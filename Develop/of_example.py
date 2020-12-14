import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#import matplotlib.pyplot as plt
from matplotlib import cm
import math



class Exp03(object):
    def __init__(self):
        print("create instance of simple OF")

        #self.video_name = "C:\\Users\\dhelm\\Documents\\1.m4v"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\1_deinterlaced.m4v"
        self.video_name = "C:\\Users\\dhelm\\Documents\\OD_Annotations\\videos\\RG600126_03142006_1456.mpg"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\OD_Annotations\\videos\\RG604314_05152007_1245.mpg"

        #
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\tilt\\tilt_130_74088.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\11_885.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\test_pan_st.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\066f929d-6434-4ea9-844b-e066f57b6c28_53.mp4"
        #self.video_name = "/data/share/maxrecall_vhh_mmsi/release/videos/downloaded/2.m4v"

        self.video_name = "C:\\Users\\dhelm\\Documents\\CMDGdataset\\CMDG dataset\\5_MagicFountain\\26_Jun_2014_19-31-32_GMT\\video-26_Jun_2014_19-31-32_GMT.m4v"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\CMDGdataset\\CMDG dataset\\5_MagicFountain\\09_Jul_2014_16-51-28_GMT\\video-09_Jul_2014_16-51-28_GMT.m4v"


    def run(self):

        frames_l = []
        mag_l = []
        ang_l = []
        n_points_l = []
        cnt = 0
        mag_sum = 0

        recording_flag = False
        start = 0  # 150  1241
        stop = 1000  # 700  1479

        cap = cv.VideoCapture(self.video_name)
        while(cap.isOpened()):
            cnt = cnt + 1
            ret, frame = cap.read()

            if (ret == False):
                break

            if(cnt >= start and cnt < stop):
                recording_flag = True
            elif(cnt >= stop):
                #recording_flag = False
                break

            if(recording_flag == True):
                frame_resized = self.crop(frame, (720, 720))
                frame_gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

                #edges = cv.Canny(frame_gray, 0, 25)

                #lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
                #if(lines is not None):
                #    print(len(lines))

                #lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
                #print(lines)

                #exit()


                #frame_gray = cv.Canny(frame_gray, 10, 20)
                kernel1 = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                kernel2 = np.array([[1, 1, 1],
                                   [1, -7, 1],
                                   [1, 1, 1]])

                #frame_gray = cv.medianBlur(frame_gray, 7)
                #gauss_kernel = cv.getGaussianKernel(7, 3.2)
                #frame_gray = cv.filter2D(frame_gray, -1, kernel2)
                frames_l.append(frame_gray)

                #frame_gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2LUV)
                #frames_l.append(frame_gray[:, :, :1])

        cap.release()
        frames_np = np.array(frames_l)

        print(frames_np.shape)


        plt.figure("asd")
        #fig.suptitle('asdfasdf')
        number_of_feature_matches = 0
        for i in range(1, len(frames_np)):
            frame_curr = frames_np[i]

            #print(number_of_features)
            if (number_of_feature_matches < 200):
                print("create new features")
                frame_prev = frames_np[i - 1]
                kp_prvs, descriptors_prvs = self.getFeatures(frame_prev)

            #print(len(kp_prvs))
            kp_curr, descriptors_curr = self.getFeatures(frame_curr)

            kp_prev_list, kp_curr_list, good_matches = self.getMatches(kp_prvs,
                                                                       descriptors_prvs,
                                                                       kp_curr,
                                                                       descriptors_curr)

            curr_points = np.array(kp_curr_list).astype('float').reshape(-1, 1, 2)
            prev_points = np.array(kp_prev_list).astype('float').reshape(-1, 1, 2)

            number_of_feature_matches = len(curr_points)
            n_points_l.append(len(curr_points))

            print(number_of_feature_matches)


            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(0, len(good_matches))]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)

            '''
            img3 = frame_curr.copy()
            img3 = cv.drawMatches(frame_prev, kp_prvs, frame_curr, kp_curr, matches1to2=good_matches, outImg=img3)
            cv.imshow("asdf", img3)
            k = cv.waitKey()
            if (k == 'n'):
                continue

            
            plt.imshow(img3, )
            plt.draw()
            plt.pause(0.2)
            '''

            if ((len(curr_points) != 0 or len(prev_points) != 0)):
                mag, ang = self.compute_magnitude_angle(prev_points, curr_points)
                #print(mag)

                mag_median = np.median(mag)
                ang_median = np.median(ang)

                mag_sum = mag_sum + mag_median

            else:
                mag_median = 0
                ang_median = 0

            mag_l.append(mag_median)
            ang_l.append(ang_median)

            mag_np = np.array(mag_l)
            ang_np = np.array(ang_l)
            n_points_np = np.array(n_points_l)

        kalman_mag_np = self.run_kalman(mag_np)
        kalman_ang_np = self.run_kalman(ang_np)

        #print(ang_np.shape)
        #print(n_points_np.shape)

        ''''''
        fig, axs = plt.subplots(3)
        axs[0].plot(np.arange(len(mag_np)), mag_np)
        axs[0].plot(np.arange(len(kalman_mag_np)), kalman_mag_np)
        axs[1].plot(np.arange(len(ang_np)), ang_np)
        axs[1].plot(np.arange(len(kalman_ang_np)), kalman_ang_np)
        axs[2].plot(np.arange(len(n_points_np)), n_points_np)
        plt.grid(True)
        plt.show()
        #plt.pause(0.002)

        print("magnitude threshold: " + str(np.median(kalman_mag_np)))
        print("angle threshold: " + str(np.median(kalman_ang_np)))

        #exit()
        '''
        #frame1_resized = frame1
        #frame1_resized = cv.resize(frame1_resized, (512, 512))
        prvs = cv.cvtColor(frame1_resized, cv.COLOR_BGR2GRAY)
        block_coordinates_np, block_center_coordinates_np = self.block_creation(prvs)

        block_kp_prvs_l = []
        block_descr_prvs_l = []
        for a in range(0, len(block_coordinates_np)):
            prvs_block = prvs[block_coordinates_np[a][0]:block_coordinates_np[a][0] + 128,
                         block_coordinates_np[a][1]:block_coordinates_np[a][1] + 128]

            kp_prvs, descriptors_prvs = self.getFeatures(prvs_block)
            block_kp_prvs_l.append(kp_prvs)
            block_descr_prvs_l.append(descriptors_prvs)

        


        fig = plt.figure()
        traj = np.zeros(shape=(600, 800, 3))
        fig, axs = plt.subplots(3)

        mag_sum = 0
        mag_sum_l = []
        prev_mag = 0
        prev_ang = 0
        n_points_l = []

        angles_l = []

        mag_l = []
        ang_l = []

        all_mag_l = []
        all_ang_l = []

        while (1):
            cnt = cnt + 1
            #print(cnt)
            ret, frame2 = cap.read()
            #print(ret)

            if cnt == 1060 or ret == False:
                break

            if(cnt >= 750):
                frame2_resized = self.crop(frame2, (720, 720))
                #frame2_resized = frame2
                #frame2_resized = cv.resize(frame2_resized, (512, 512))
                next = cv.cvtColor(frame2_resized, cv.COLOR_BGR2GRAY)

                block_coordinates_np, block_center_coordinates_np = self.block_creation(next)
                # print(block_coordinates_np)
                # print(block_center_coordinates_np)

                # calculate for each block mag and ang

                block_kp_curr_l = []
                block_descr_curr_l = []
                for a in range(0, len(block_coordinates_np)):
                    curr_block = next[block_coordinates_np[a][0]:block_coordinates_np[a][0] + 128,
                                 block_coordinates_np[a][1]:block_coordinates_np[a][1] + 128]

                    kp_curr, descriptors_curr = self.getFeatures(curr_block)
                    block_kp_curr_l.append(kp_curr)
                    block_descr_curr_l.append(descriptors_curr)

                block_kp_prev_list = []
                block_kp_curr_list = []
                for e in range(0, len(block_descr_curr_l)):
                    #print(kp_curr)
                    kp_prev_list, kp_curr_list = self.getMatches(block_kp_prvs_l[e],
                                                                 block_descr_prvs_l[e],
                                                                 block_kp_curr_l[e],
                                                                 block_descr_curr_l[e])

                    curr_points = np.array(kp_curr_list).astype('float').reshape(-1, 1, 2)
                    prev_points = np.array(kp_prev_list).astype('float').reshape(-1, 1, 2)
                    block_kp_prev_list.append(prev_points)
                    block_kp_curr_list.append(curr_points)

                    #print(len(kp_prev_list))
                    #print(len(kp_curr_list))

                block_mag_l = []
                block_ang_l = []
                for blk_id in range(0, len(block_kp_curr_list)):
                    blk_curr_points = block_kp_curr_list[blk_id]
                    blk_prev_points = block_kp_prev_list[blk_id]

                    print(len(block_kp_curr_list[blk_id]))
                    print(len(block_kp_prev_list[blk_id]))

                    MIN_FEAT = 1
                    if ((len(blk_curr_points) != 0 or len(blk_prev_points) != 0) and len(blk_curr_points) >= MIN_FEAT):
                        block_mag, block_ang = self.compute_magnitude_angle(blk_prev_points, blk_curr_points)
                    else:
                        block_mag = 0
                        block_ang = 0
                    block_mag_l.append(np.median(block_mag))
                    block_ang_l.append(np.median(block_ang))

                block_mag_np = np.array(block_mag_l)
                block_ang_np = np.array(block_ang_l)

                mag_l.append(block_mag_np)
                ang_l.append(block_ang_np)
                mag_np = np.array(mag_l)
                ang_np = np.array(ang_l)

                print(mag_np.shape)
                print(ang_np.shape)
                #continue


                fig.suptitle('asdfasdf')
                pos = 2
                axs[0].plot(np.arange(len(mag_np)), mag_np)
                axs[1].plot(np.arange(len(ang_np)), ang_np)
                #axs[2].plot(np.arange(len(angles_np)), angles_np[:, 2:3])
                plt.grid(True)
                plt.draw()
                plt.pause(0.002)
            #plt.show()
            #exit()

            #MIN_FEAT = 10

            ## do magic staff

            #if ((len(curr_points) != 0 or len(prev_points) != 0) and len(kp_curr_list) >= MIN_FEAT):

            #print("calculate mag")
            #mag, ang = self.compute_magnitude_angle(prev_points, curr_points)
            #prev_mag = mag
            #prev_ang = ang
            '''
        '''
            u, v = cv.polarToCart(np.array(mag), np.array(ang), angleInDegrees=True)
            filtered_u, outliers = self.filter1D(u, alpha=1.9)
            filtered_v, outliers = self.filter1D(v, alpha=1.9)
            #s_u = np.max(np.abs(filtered_u))
            #s_v = np.max(np.abs(filtered_v))
            #s = max(s_u, s_v)

            
            E, mask_e = cv.findEssentialMat(prev_points, curr_points, focal=1.0, pp=(0., 0.),
                                             method=cv.RANSAC, prob=0.999, threshold=1.0)

            #print("Essential matrix: used ", np.sum(mask_e), " of total ", len(prev_points), "matches")

            points, R, t, mask_RP = cv.recoverPose(E, prev_points, curr_points, mask=mask_e)
            #print("points:", points, "\trecover pose mask:", np.sum(mask_RP != 0))
            #print("R:", R, "t:", t.T)

            angles = self.rotationMatrixToEulerAngles(R)
            angles = np.rad2deg(angles)

            print(angles)
            angles_l.append(angles)
            angles_np = np.array(angles_l)
            print(np.array(angles_l).shape)

            if(cnt >= 3):

                fig.suptitle('asdfasdf')
                axs[0].plot(np.arange(len(angles_np)), angles_np[:, :1])
                axs[1].plot(np.arange(len(angles_np)), angles_np[:, 1:2])
                axs[2].plot(np.arange(len(angles_np)), angles_np[:, 2:3])
                plt.grid(True)
                plt.draw()
                plt.pause(0.002)


            #exit()
            diag = np.array([[-1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]])
            mono_coord = np.matmul(diag, t).flatten()
            #print(mono_coord)

            draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
            traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

            cv.imshow('trajectory', traj)
            k = cv.waitKey(20)
            if k == 27:
                break

            #exit()

            bool_mask = mask_RP.astype(bool)
            
            # Create 3 x 4 Homogenous Transform
            Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
            print("Pose_1: ", Pose_1)
            Pose_2 = np.hstack((R, t))
            print("Pose_2: ", Pose_2)

            # Points Given in N,1,2 array
            landmarks_hom = cv.triangulatePoints(Pose_1, Pose_2,
                                                  prev_points[mask_RP[:, 0] == 1],
                                                  curr_points[mask_RP[:, 0] == 1]).T
            landmarks_hom_norm = landmarks_hom / landmarks_hom[:, -1][:, None]
            landmarks = landmarks_hom_norm[:, :3]


            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('equal')  # important!
            title = ax.set_title('3D Test')
            ax.set_zlim3d(-5, 10)

            # Plot triangulated featues in Red
            #graph, = ax.plot(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], linestyle="", marker="o",
            #                 color='r')

            # Plot pose 1
            self.plot_pose3_on_axes(ax, np.eye(3), np.zeros(3)[np.newaxis], axis_length=2.0)
            # Plot pose 2
            self.plot_pose3_on_axes(ax, R, t.T, axis_length=2.0)
            ax.set_zlim3d(-3, 3)
            ax.set_xlim3d(-3, 3)
            ax.set_ylim3d(-3, 3)
            ax.view_init(-70, -90)
            #plt.show()
            plt.draw()
            plt.pause(0.002)
        
            #exit()

            print(np.median(mag))
            print(np.median(ang))
            print(np.mean(mag))
            print(np.mean(ang))
            print(np.std(mag))
            print(np.std(ang))

            fil_mag, outliers_idx = self.filter1D(mag, alpha=1.5)
            fil_mag, outliers_idx = self.filter1D(fil_mag, alpha=1.5)
            fil_mag, outliers_idx = self.filter1D(fil_mag, alpha=1.3)

            fil_ang, outliers_idx = self.filter1D(ang, alpha=1.5)
            fil_ang, outliers_idx = self.filter1D(fil_ang, alpha=1.5)
            fil_ang, outliers_idx = self.filter1D(fil_ang, alpha=1.3)
            print("aa")
            #print(np.median(fil_mag))
            #print(np.median(ang))
            #print(np.mean(fil_mag))
            #print(np.mean(ang))
            #print(np.std(fil_mag))
            #print(np.std(ang))



            #plt.figure()
            #plt.plot(np.arange(len(fil_mag)), fil_mag)
            #plt.show()

            mag_l.append(np.median(mag))
            ang_l.append(np.median(ang))
            #mag_l.append(np.median(fil_mag))
            #ang_l.append(np.median(fil_ang))
        elif(len(kp_curr_list) < 10):
            mag = 0
            ang = 0
            mag_l.append(np.median(prev_mag))
            ang_l.append(np.median(prev_ang))


        if (len(kp_curr_list) < MIN_FEAT):
            kp_prevs, descriptors_prev = self.getFeatures(next)

        #xit()
        prvs = next
        #exit()
        '''

        '''
        #print(len(kp_prev_list))
        #print(len(kp_curr_list))
        curr_points = np.array(kp_curr_list).astype('float').reshape(-1, 1, 2)
        prev_points = np.array(kp_prev_list).astype('float').reshape(-1, 1, 2)
        #print(len(curr_points))
        #print(len(prev_points))

        if(len(curr_points) != 0 or len(prev_points) != 0):
            mag, ang = self.compute_magnitude_angle(prev_points, curr_points)

            u, v = cv.polarToCart(np.array(mag), np.array(ang), angleInDegrees=True)
            filtered_u, outliers = self.filter1D(u, alpha=0.9)
            filtered_v, outliers = self.filter1D(v, alpha=0.9)
            s_u = np.max(np.abs(filtered_u))
            s_v = np.max(np.abs(filtered_v))
            s = max(s_u, s_v)

            if (s == 0):
                s = 4
            #print(s)


            filtered_block_ang, outliers = self.filter1D(mag, alpha=0.9)
            filtered_block_mag, outliers = self.filter1D(np.abs(ang), alpha=0.9)
            #print(filtered_block_mag.shape)
            
        
            
            plt.hist2d(filtered_block_ang, filtered_block_mag, bins=10, range=[[0, 20], [0, 360]])
            plt.draw()
            plt.pause(0.0002)
            ''''''
        else:
            mag = 0
            ang = 0
        #print(mag)
        
        mag_l.append(np.median(mag))
        ang_l.append(np.median(ang))
        
        exit()

        mag_np = np.array(mag_l)
        ang_np = np.array(ang_l)
        n_points_np = np.array(n_points_l)


        # zi = (xi − min(x) ) / ( max(x) − min(x) )

        mag_min = np.min(mag_np)
        mag_max = np.max(mag_np)
        norm_mag_np = (mag_np - mag_min ) / ((mag_max - mag_min) + 0.000000001)

        mag_min = np.min(n_points_np)
        mag_max = np.max(n_points_np)
        norm_n_points_np = (n_points_np - mag_min) / ((mag_max - mag_min) + 0.000000001)

        '''


        #norm = np.linalg.norm(n_points_np)
        #norm_n_points_np = n_points_np / norm

        #norm = np.linalg.norm(mag_np)
        #norm_mag_np = mag_np / norm

        '''
        from scipy.ndimage.filters import uniform_filter1d, gaussian_filter1d
        N = 15
        filtered_mag_np = uniform_filter1d(mag_np, size=N)
        filtered_ang_np = uniform_filter1d(ang_np, size=N)
        filtered_n_np = uniform_filter1d(n_points_np, size=N)

        #kalman_mag_np = self.run_kalman(mag_np)
        #kalman_ang_np = self.run_kalman(ang_np)

        #filtered_mag_np, outliers_idx = self.filter1D(norm_mag_np, alpha=0.8)
        #filtered_ang_np, outliers_idx = self.filter1D(ang_np, alpha=1.2)
        #filtered_n_np, outliers_idx = self.filter1D(norm_n_points_np, alpha=1.2)


        # plot angles over time (frames)
        fig, axs = plt.subplots(4)
        fig.suptitle('mag and angles per feature point in one frame')



        #axs[0].plot(np.arange(len(mag_np)), mag_np)
        #axs[0].plot(np.arange(len(mag_np)), mag_np)
        axs[0].plot(np.arange(len(filtered_mag_np)), filtered_mag_np)
        #axs[0].plot(np.arange(len(kalman_mag_np)), kalman_mag_np)

        #axs[1].plot(np.arange(len(ang_np)), ang_np)
        axs[1].plot(np.arange(len(filtered_ang_np)), filtered_ang_np)
        #axs[1].plot(np.arange(len(kalman_ang_np)), kalman_ang_np)
        #        b, bins, patches = axs[2].hist(ang_np, bins=8, range=[0,360], cumulative=False)  #bins=None, range=None
        axs[2].plot(np.arange(len(mag_sum_l)), mag_sum_l)

        axs[3].plot(np.arange(len(filtered_n_np)), np.array(filtered_n_np))
        #plt.ylim(ymax=190, ymin=-190)
        plt.grid(True)
        plt.show()
        '''
        '''        '''
    def block_creation(self, frame_np):
        # split frame in nxm tiles
        grid_x = 4
        grid_y = 4
        h = frame_np.shape[0]
        w = frame_np.shape[1]

        #print(frame_np.shape)

        #print(w)
        #print(h)

        if (w % grid_x == 0):
            block_size_x = int(w / grid_x)
        else:
            print("not valid grid number")
            exit()
        if (h % grid_y == 0):
            block_size_y = int(h / grid_y)
        else:
            print("not valid grid number")
            exit()

        block_coordinates_l = []
        block_center_coordinates_l = []
        for b_x in range(0, grid_x):
            for b_y in range(0, grid_y):
                #print("----")
                #print(str(b_x) + str(b_y))
                #print(str(b_x * block_size_x) + "|" + str(b_x * block_size_x + block_size_x))
                #print(str(b_y * block_size_y) + "|" + str(b_y * block_size_y + block_size_y))
                # print("----")
                # print(str(b_x) + str(b_y))
                # print(str(b_x * block_size_x) + "|" + str(b_x * block_size_x + block_size_x))
                # print(str(b_y * block_size_y) + "|" + str(b_y * block_size_y + block_size_y))

                block_coordinates_l.append([b_x * block_size_x, b_y * block_size_y])

                #print(str(b_x * block_size_x / 2) + "|" + str((b_x * block_size_x + block_size_x) / 2))
                #print(str(b_y * block_size_y / 2) + "|" + str((b_y * block_size_y + block_size_y) / 2))
                # print(str(b_x * block_size_x / 2) + "|" + str((b_x * block_size_x + block_size_x) / 2))
                # print(str(b_y * block_size_y / 2) + "|" + str((b_y * block_size_y + block_size_y) / 2))
                block_center_coordinates_l.append([b_x * block_size_x + block_size_x / 2,
                                                   b_y * block_size_y + block_size_x / 2])

        block_coordinates_np = np.array(block_coordinates_l).astype('int')
        block_center_coordinates_np = np.array(block_center_coordinates_l).astype('int')
        #print(block_coordinates_np)
        #print(block_center_coordinates_np)

        return block_coordinates_np, block_center_coordinates_np

    def plot_pose3_on_axes(self, axes, gRp, origin, axis_length=0.1):
        """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
        # get rotation and translation (center)
        # gRp = pose.rotation().matrix()  # rotation from pose to global
        # t = pose.translation()
        # origin = np.array([t.x(), t.y(), t.z()])

        # draw the camera axes
        x_axis = origin + gRp[:, 0] * axis_length
        line = np.append(origin, x_axis, axis=0)
        axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

        y_axis = origin + gRp[:, 1] * axis_length
        line = np.append(origin, y_axis, axis=0)
        axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

        z_axis = origin + gRp[:, 2] * axis_length
        line = np.append(origin, z_axis, axis=0)
        axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')

    def rotationMatrixToEulerAngles(self, R):

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])


    def multiply(self, mu1, var1, mu2, var2):
        if var1 == 0.0:
            var1 = 1.e-80
        if var2 == 0:
            var2 = 1e-80

        mean = (var1 * mu2 + var2 * mu1) / (var1 + var2)
        variance = 1 / (1 / var1 + 1 / var2)
        return (mean, variance)

    def run_kalman(self, signal_1d):
        ##### assume dog is always moving 1m to the right
        np.random.seed(13)
        pos = (0., 400.)  # gaussian N(0, 400)
        velocity = 1.
        # variance in process model and the RFID sensor
        process_variance = 0.2 # 0.5
        sensor_variance = 1.
        N = len(signal_1d)

        positions = np.zeros(N)
        for i, z in enumerate(signal_1d):
            pos = self.predict(pos=pos[0],
                               variance=pos[1],
                               movement=velocity,
                               movement_variance=process_variance)
            #print('PREDICT: {: 10.4f} {: 10.4f}'.format(pos[0], pos[1]), end='\t')

            pos = self.update(mean=pos[0],
                              variance=pos[1],
                              measurement=z,
                              measurement_variance=sensor_variance)
            positions[i] = pos[0]
            #print('UPDATE: {: 10.4f} {: 10.4f}\tZ: {:.4f}'.format(pos[0], pos[1], z))

        return positions

    def gmm(self, X):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=4).fit(X)
        labels = gmm.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

    def predict(self, pos, variance, movement, movement_variance):
        return (pos + movement, variance + movement_variance)

    def update(self, mean, variance, measurement, measurement_variance):
        return self.multiply(mean, variance, measurement, measurement_variance)

    def f(self, mu, sigma2, x):
        coef = 1 / math.sqrt(2.0 * math.pi * sigma2)
        expn = math.exp(-0.5 * (x - mu) ** 2 / sigma2)
        return coef * expn

    def getFeatures(self, frame):
        kp_curr_list = []
        kp_prev_list = []
        '''
        detector = cv.xfeatures2d.SIFT_create()
        '''
        detector = cv.ORB_create(nfeatures=1000,
                                 scaleFactor=1.0,
                                 nlevels=2,
                                 edgeThreshold=15,
                                 firstLevel=0,
                                 WTA_K=2,
                                 scoreType=cv.ORB_FAST_SCORE,  #cv2.ORB_HARRIS_SCORE  ORB_FAST_SCORE
                                 patchSize=31,
                                 fastThreshold=10,
                                 )

        #nfeatures=None, scaleFactor=None, nlevels=None, edgeThreshold=None, firstLevel=None, WTA_K=None, scoreType=None, patchSize=None, fastThreshold=None
        kp, descriptors = detector.detectAndCompute(frame, None)

        return kp, descriptors

    def getMatches(self, kp1, descriptor1, kp2, descriptor2):
        kp_curr_list = []
        kp_prev_list = []

        matcher_type = "flann"
        if(matcher_type == "bf"):

            try:
                # Create a Brute Force Matcher object.
                bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)  #, VTA_K=3
                # Perform the matching between the ORB descriptors of the training image and the test image
                matches = bf.match(descriptor1, descriptor2)
            except:
                return kp_prev_list, kp_curr_list

                # print(matches)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(0, len(matches))]


            # normalize
            distances_l = []
            for i in range(0, len(matches)):
                m = matches[i]
                #print(m.distance)
                distances_l.append(m.distance)

            distances_np = np.array(distances_l)
            max_val = np.max(distances_np)
            min_val = np.max(distances_np)

            good_matches = []
            for i in range(0, len(matches)):
                m = matches[i]
                #print((m.distance / max_val) * 1)
                if (m.distance / max_val) * 1 < 0.45:
                    good_matches.append(m)

        elif(matcher_type == "flann"):
            # FLANN parameters   ### FLANN_INDEX_LSH=6  FLANN_INDEX_KDTREE = 0
            FLANN_INDEX_LSH=6
            index_params = dict(algorithm=FLANN_INDEX_LSH,  # FLANN_INDEX_LSH=6
                                table_number=2,  # 12
                                key_size=5,  # 20
                                multi_probe_level=5)  # 2
            search_params = dict(checks=500)  # or pass empty dictionary

            #print(descriptor1)
            #print(descriptor2)

            try:
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(descriptor1, descriptor2, k=2)
                #print(matches)
            except:
                return kp_prev_list, kp_curr_list

            #print(matches)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(0, len(matches))]

            # ratio test as per Lowe's paper
            good_matches = []
            for i in range(0, len(matches)):
            #for i, (m, n) in enumerate(matches):
                ##print(i)
                #print(matches)
                if(len(matches[i]) != 2):
                    continue

                m = matches[i][0]
                n = matches[i][1]

                if m.distance < 0.35 * n.distance:
                    good_matches.append(m)

        elif (matcher_type == "bfknn"):

            try:
                bf = cv.BFMatcher()
                matches = bf.knnMatch(descriptor1, descriptor2, k=2)

            except:
                return kp_prev_list, kp_curr_list

            # print(matches)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(0, len(matches))]

            # ratio test as per Lowe's paper
            good_matches = []
            for i in range(0, len(matches)):
                # for i, (m, n) in enumerate(matches):
                ##print(i)
                # print(matches)
                if (len(matches[i]) != 2):
                    continue

                m = matches[i][0]
                n = matches[i][1]

                if m.distance < 0.15 * n.distance:
                    good_matches.append(m)

        # The matches with shorter distance are the ones we want.
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        for match in good_matches:
            kp_curr_list.append(kp2[match.trainIdx].pt)
            kp_prev_list.append(kp1[match.queryIdx].pt)

        #MIN_MATCHES = 50
        #if len(good_matches) > MIN_MATCHES:
        #    src_points = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #    dst_points = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return kp_prev_list, kp_curr_list, good_matches

    def filter2D(self, data_np, sigma=5.):
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(data_np, sigma=sigma)
        return result

    def filter1D(self, data_np, alpha=2.):
        #print(type(data_np))
        #print(data_np.shape)

        data_std = np.std(data_np)
        data_mean = np.mean(data_np)
        anomaly_cut_off = data_std * alpha

        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off
        #print(lower_limit)
        # Generate outliers
        outliers_idx = []
        for o, outlier in enumerate(data_np):
            if outlier > upper_limit or outlier <= lower_limit:
                outliers_idx.append(o)


        filtered_data_np = data_np.copy()
        for j in range(0, len(outliers_idx)):
            if (outliers_idx[j] + 1 >= len(data_np)):
                break

            prev_val = data_np[outliers_idx[j] - 1]
            next_val = data_np[outliers_idx[j] + 1]
            mean_val = (prev_val + next_val) / 2

            filtered_data_np[outliers_idx[j]] = mean_val
        ''''''
        '''
        filtered_data_np = np.delete(data_np, outliers_idx)
        '''
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
        #print(d.shape)
        mag = np.hypot(d[:, 0, 0], d[:, 0, 1])
        ang = np.abs(np.round(np.degrees(np.arctan2(d[:, 0, 1], d[:, 0, 0]))))

        return mag, ang


class SimpleOF(object):
    def __init__(self):
        print("create instance of simple OF")

        self.video_name = "C:\\Users\\dhelm\\Documents\\1.m4v"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\tilt\\tilt_130_74088.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\11_885.mp4"
        #self.video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\20_3767.mp4"

    def run(self):
        import numpy as np
        import cv2 as cv

        mag_l = []
        ang_l = []
        cnt = 0

        cap = cv.VideoCapture(self.video_name)
        ret, frame1 = cap.read()
        frame1_resized = self.crop(frame1, (720, 720))
        frame1_resized = cv.resize(frame1_resized, (512, 512))
        prvs = cv.cvtColor(frame1_resized, cv.COLOR_BGR2GRAY)
        #prvs = cv.medianBlur(prvs, 3)


        #prvs = cv.Canny(prvs, 20, 50)
        hsv = np.zeros_like(frame1_resized)
        hsv[..., 1] = 255

        plt.figure()
        #plt.figure(0)
        #plt.ion()
        #ax = plt.gca(projection='3d')
        #plt.suptitle('mag surf')
        #plt.suptitle('ang surf')
        while (1):
            cnt = cnt + 1
            ret, frame2 = cap.read()
            if cnt == 1000 or ret == False:
                break
            frame2_resized = self.crop(frame2, (720, 720))
            frame2_resized = cv.resize(frame2_resized, (512, 512))
            next = cv.cvtColor(frame2_resized, cv.COLOR_BGR2GRAY)
            #next = cv.medianBlur(next, 9)

            #next = self.filter2D(next, sigma=10)

            #next = cv.Canny(next, 20, 50)

            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 10, 7, 5, 1.2, 0)   # 7 .. 1.5
            # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            print(flow.shape)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            print(mag[0][0])
            print(cnt)

            print(np.mean(mag))
            print(np.std(mag))

            '''
            #mag = np.zeros((mag.shape))
            #mag[100][100] = cnt
            #print(mag[0][0])


            # Make data.
            X = np.arange(0, flow.shape[0], 1)
            Y = np.arange(0, flow.shape[1], 1)
            X, Y = np.meshgrid(X, Y)
            Z = mag

            # Plot the surface.
            ax.plot_surface(X, Y, Z, linewidth=1, antialiased=True, cmap=cm.jet)
            #ax.relim()  # Recalculate limits
            #ax.autoscale_view(True, True, True)  # Autoscale

            plt.draw()
            plt.pause(0.002)
            ax.cla()

            prvs = next
            continue
            '''

            ''''''
            block_coordinates_np, block_center_coordinates_np = self.block_creation(frame2_resized)
            #print(block_coordinates_np)
            #print(block_center_coordinates_np)

            # calculate for each block mag and ang

            block_mag_l = []
            block_ang_l = []
            vector_flow = []
            for a in range(0, len(block_coordinates_np)):
                #print("###")
                #print(block_coordinates_np[a])

                mag_block = mag[block_coordinates_np[a][0]:block_coordinates_np[a][0]+128,
                                block_coordinates_np[a][1]:block_coordinates_np[a][1]+128]
                block_mag_l.append(np.median(mag_block))
                #print(mag_block.shape)

                ang_block = ang[block_coordinates_np[a][0]:block_coordinates_np[a][0]+128,
                                block_coordinates_np[a][1]:block_coordinates_np[a][1]+128]
                block_ang_l.append(np.median(ang_block))
                #print(ang_block.shape)
                #print(np.median(mag_block))
                #x,  = cv.polarToCart(np.median(mag_block), np.median(ang_block), angleInDegrees=True)

                vector_x = np.multiply(np.mean(mag_block), np.cos(np.deg2rad(np.mean(ang_block))))
                vector_y = np.multiply(np.mean(mag_block), np.sin(np.deg2rad(np.mean(ang_block))))
                vector_flow.append([vector_x, vector_y])
                #print(vector_flow)

            #print(block_mag_l)
            #print(block_ang_l)

            #print(vector_flow)

            '''
            for a in range(0, len(block_center_coordinates_np)):
                #print(block_center_coordinates_np[a])
                #print(vector_flow[a])

                cv.circle(frame2_resized,
                          center=(block_center_coordinates_np[a][0], block_center_coordinates_np[a][1]),
                          radius=3,
                          color=(255, 0, 0),
                          thickness=2
                          )
                cv.line(frame2_resized,
                        pt1=(block_center_coordinates_np[a][0],
                             block_center_coordinates_np[a][1]),
                        pt2=(block_center_coordinates_np[a][0]+int(vector_flow[a][0]*2),
                             block_center_coordinates_np[a][1] + int(vector_flow[a][1] * 2)),
                        color=(255, 0, 0),
                        thickness=2)
                cv.line(frame2_resized,
                        pt1=(block_center_coordinates_np[a][0],
                             block_center_coordinates_np[a][1]),
                        pt2=(block_center_coordinates_np[a][0],
                             block_center_coordinates_np[a][1] + int(vector_flow[a][1] * 2)),
                        color=(0, 255, 0),
                        thickness=2)

            
            hsv = np.zeros_like(frame1_resized)
            hsv[..., 1] = 255
            hsv[..., 0] = ang #* 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame2', bgr)

            cv.imshow('frame', frame2_resized)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            '''
            # summarize block results --> result per frame
            tmp_mag = np.median(np.array(block_mag_l))
            tmp_ang = np.median(np.array(block_ang_l))

            mag_l.append(tmp_mag)
            ang_l.append(tmp_ang)

            u, v = cv.polarToCart(np.array(block_mag_l), np.array(block_ang_l), angleInDegrees=True)
            s_u = np.max(np.abs(u))
            s_v = np.max(np.abs(v))
            s = max(s_u, s_v)

            if (s == 0):
                s = 4
            print(s)

            filtered_block_ang, outliers = self.filter1D(np.array(block_ang_l), alpha=0.1)
            filtered_block_mag, outliers = self.filter1D(np.array(block_mag_l), alpha=0.1)
            print(filtered_block_ang)

            '''
            plt.hist2d(filtered_block_mag, filtered_block_ang, bins=47, range=[[0,10], [0,360]])
            plt.draw()
            plt.pause(0.002)
            '''

            prvs = next




        print(mag_l)
        print(len(mag_l))

        print(np.array(mag_l))
        print(np.array(mag_l).shape)
        print(np.mean(np.array(mag_l), axis=0))

        final_mag = np.array(mag_l)
        final_ang = np.array(ang_l)

        from scipy.ndimage.filters import uniform_filter1d, gaussian_filter1d
        N = 15
        filtered_mag_np = uniform_filter1d(final_mag, size=N)
        filtered_ang_np = uniform_filter1d(final_ang, size=N)


        fig, axs = plt.subplots(2)
        fig.suptitle('asd')

        # axs[0].plot(np.arange(len(mag_np)), mag_np)
        # axs[0].plot(np.arange(len(mag_np)), mag_np)
        axs[0].plot(np.arange(len(filtered_mag_np)), filtered_mag_np)
        axs[1].plot(np.arange(len(filtered_ang_np)), filtered_ang_np)
        plt.show()
        exit()

        plt.figure()
        plt.hist2d(final_mag, final_ang)
        plt.show()

        exit()

        #final_mag = np.mean(np.array(mag_l), axis=0)
        #final_ang = np.mean(np.array(ang_l), axis=0)

        #b, bins, patches = plt.hist(block_ang_l, bins=8, range=[0, 360],
        #                            cumulative=False)  # bins=None, range=None

        #filtered_ang_np, outliers_idx = self.filter1D(np.array(ang_l), alpha=1)
        #filtered_ang_np1, outliers_idx = self.filter1D(np.array(ang_l), alpha=3)
        filtered_ang_np, outliers_idx = self.filter1D(final_ang, alpha=0.1)
        #filtered_ang_np, outliers_idx = self.filter1D(filtered_ang_np, alpha=5)

        filtered_mag_np, outliers_idx = self.filter1D(final_mag, alpha=0.1)

        # plot angles over time
        fig, axs = plt.subplots(3)
        fig.suptitle('mag and angles')

        axs[0].plot(np.arange(len(final_mag)), final_mag) #, color='k')
        axs[0].plot(np.arange(len(filtered_mag_np)), filtered_mag_np)  # , color='k')
        axs[0].set_ylim([0, 5])

        axs[1].plot(np.arange(len(final_ang)), final_ang) #, color='y')
        axs[1].plot(np.arange(len(filtered_ang_np)), filtered_ang_np)  # , color='k')
        axs[1].set_ylim([-90, 360])

        axs[2].hist(final_ang, bins=8, range=[0, 360], cumulative=False)
        plt.grid(True)
        plt.show()

        print("FINISHED")

    def block_creation(self, frame_np):
        # split frame in nxm tiles
        grid_x = 4
        grid_y = 4
        h = frame_np.shape[0]
        w = frame_np.shape[1]

        #print(frame_np.shape)

        #print(w)
        #print(h)

        if (w % grid_x == 0):
            block_size_x = int(w / grid_x)
        else:
            print("not valid grid number")
            exit()
        if (h % grid_y == 0):
            block_size_y = int(h / grid_y)
        else:
            print("not valid grid number")
            exit()

        block_coordinates_l = []
        block_center_coordinates_l = []
        for b_x in range(0, grid_x):
            for b_y in range(0, grid_y):
                #print("----")
                #print(str(b_x) + str(b_y))
                #print(str(b_x * block_size_x) + "|" + str(b_x * block_size_x + block_size_x))
                #print(str(b_y * block_size_y) + "|" + str(b_y * block_size_y + block_size_y))
                # print("----")
                # print(str(b_x) + str(b_y))
                # print(str(b_x * block_size_x) + "|" + str(b_x * block_size_x + block_size_x))
                # print(str(b_y * block_size_y) + "|" + str(b_y * block_size_y + block_size_y))

                block_coordinates_l.append([b_x * block_size_x, b_y * block_size_y])

                #print(str(b_x * block_size_x / 2) + "|" + str((b_x * block_size_x + block_size_x) / 2))
                #print(str(b_y * block_size_y / 2) + "|" + str((b_y * block_size_y + block_size_y) / 2))
                # print(str(b_x * block_size_x / 2) + "|" + str((b_x * block_size_x + block_size_x) / 2))
                # print(str(b_y * block_size_y / 2) + "|" + str((b_y * block_size_y + block_size_y) / 2))
                block_center_coordinates_l.append([b_x * block_size_x + block_size_x / 2,
                                                   b_y * block_size_y + block_size_x / 2])

        block_coordinates_np = np.array(block_coordinates_l).astype('int')
        block_center_coordinates_np = np.array(block_center_coordinates_l).astype('int')
        #print(block_coordinates_np)
        #print(block_center_coordinates_np)

        return block_coordinates_np, block_center_coordinates_np

        # print(block_coordinates_np)
        # print(block_center_coordinates_np)

    def filter2D(self, data_np, sigma=5.):
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(data_np, sigma=sigma)
        return result

    def filter1D(self, data_np, alpha=2.):
        # print(type(data_np))
        # print(data_np.shape)

        data_std = np.std(data_np)
        data_mean = np.mean(data_np)
        anomaly_cut_off = data_std * alpha

        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off
        # print(lower_limit)
        # Generate outliers
        outliers_idx = []
        for o, outlier in enumerate(data_np):
            if outlier > upper_limit or outlier < lower_limit:
                outliers_idx.append(o)

        ''''''
        filtered_data_np = data_np.copy()
        for j in range(0, len(outliers_idx)):
            if (outliers_idx[j] + 1 >= len(data_np)):
                break

            prev_val = data_np[outliers_idx[j] - 1]
            next_val = data_np[outliers_idx[j] + 1]
            mean_val = (prev_val + next_val) / 2

            filtered_data_np[outliers_idx[j]] = mean_val

        '''
        filtered_data_np = np.delete(data_np, outliers_idx)
        '''
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


def testStabilization():
    # import required libraries
    from vidgear.gears import VideoGear
    import numpy as np
    import cv2

    #video_name = "C:\\Users\\dhelm\\Documents\\1.m4v"
    #video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\tilt\\tilt_130_74088.mp4"
    video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\11_885.mp4"
    #video_name = "C:\\Users\\dhelm\\Documents\\test_pan_st.mp4"
    #video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\066f929d-6434-4ea9-844b-e066f57b6c28_53.mp4"

    # open any valid video stream with stabilization enabled(`stabilize = True`)
    stream_org = VideoGear(source=video_name, stabilize=False).start()
    stream_stab = VideoGear(source=video_name, stabilize=True).start()

    # loop over
    while True:

        # read stabilized frames
        frame_stab = stream_stab.read()

        # check for stabilized frame if None-type
        if frame_stab is None:
            break

        # read un-stabilized frame
        frame_org = stream_org.read()

        # concatenate both frames
        output_frame_ = np.concatenate((frame_org, frame_stab), axis=1)
        output_frame = cv2.resize(output_frame_, (int(output_frame_.shape[1]/2), int(output_frame_.shape[0]/2)))

        # put text over concatenated frame
        cv2.putText(
            output_frame, "Before", (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2,
        )
        cv2.putText(
            output_frame, "After", (output_frame.shape[1] // 2 + 10, output_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2,
        )

        # Show output window
        cv2.imshow("Stabilized Comparison", output_frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close both video streams
    stream_org.stop()
    stream_stab.stop()

def test():
    #video_name = "C:\\Users\\dhelm\\Documents\\1.m4v"
    #video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\tilt\\tilt_130_74088.mp4"
    video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\11_885.mp4"
    # video_name = "C:\\Users\\dhelm\\Documents\\test_pan_st.mp4"
    # video_name = "C:\\Users\\dhelm\\Documents\\training_data_patrick_link_reworked\\training_data\\pan\\066f929d-6434-4ea9-844b-e066f57b6c28_53.mp4"

    from vidstab import VidStab
    import matplotlib.pyplot as plt

    stabilizer = VidStab()
    stabilizer.stabilize(input_path=video_name, output_path="./stable_video.avi", border_type='replicate')

    stabilizer.plot_trajectory()
    plt.show()

    stabilizer.plot_transforms()
    plt.show()

of = Exp03()
of.run()

#test()

