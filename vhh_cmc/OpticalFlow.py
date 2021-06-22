import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from vhh_cmc.OpticalFlow_ORB import OpticalFlow_ORB
from vhh_cmc.OpticalFlow_SIFT import OpticalFlow_SIFT
from vhh_cmc.OpticalFlow_Dense import OpticalFlow_Dense



class OpticalFlow(object):
    def __init__(self, video_frames=None, algorithm="orb", config_instance=None):
        self.video_frames = video_frames
        self.config_instance = config_instance

        self.number_of_blocks = 32

        if (algorithm == "sift"):
            self.feature_detector = OpticalFlow_SIFT(video_frames=video_frames)
        elif (algorithm == "orb"):
            self.feature_detector = OpticalFlow_ORB(video_frames=video_frames)
        else:
            print("ERROR: select valid feature extractor [e.g. sift, orb, pesc]")
            exit()

    def outlier_detection_db_scan(self, mag, ang):
        size = 5000

        data1 = np.expand_dims(mag.flatten()[:size], axis=0)
        data2 = np.expand_dims(np.radians(ang.flatten())[:size], axis=0)
        print(data1)
        print(data2)

        #data = np.zeros(512, 2)
        data = np.concatenate((data1, data2), axis=0)
        print(data.shape)

        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=0.8, min_samples=10).fit(data)
        # Scatter plot function
        colors = model.labels_
        print(colors)
        plt.scatter(data[:, 0], data[:, 1], c=colors, marker='o')
        #plt.xlabel('Concentration of flavanoids', fontsize=16)
        #plt.ylabel('Color intensity', fontsize=16)
        #plt.title('Concentration of flavanoids vs Color intensity', fontsize=20)
        plt.show()

        exit()

    def outlier_detection_2d_array(self, data):

        data1 = data[:, 1]
        data2 = data[:, 2]

        plt.scatter(data1, data2)

    def runDense(self):
        frames_np = np.squeeze(self.video_frames)

        of_dense_instance = OpticalFlow_Dense()
        hsv = np.zeros_like(frames_np[0])
        hsv[..., 1] = 255

        angles_l_n = []
        mag_l_n = []

        new_blocks_mag_l = []
        new_blocks_ang_l = []

        all_frame_u_mean_l = []
        all_frame_v_mean_l = []
        all_frame_mag_mean_l = []
        all_frame_ang_mean_l = []
        step_size = 1

        #fig = plt.figure()
        #gs = gridspec.GridSpec(nrows=1, ncols=2)

        orig_mag_l = []
        orig_angles_l = []
        orig_u_l = []
        orig_v_l = []

        if(self.config_instance.save_debug_pkg_flag == True):
            frame_size = (frames_np[0].shape[1], frames_np[0].shape[0])
            self.video_writer = cv2.VideoWriter("/data/share/dense_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12,
                                                frame_size)

        for i in range(step_size, len(frames_np), step_size):
            prev_frame = frames_np[i - step_size]
            curr_frame = frames_np[i]

            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            prev_frame_equ = cv2.equalizeHist(prev_frame)
            curr_frame_equ = cv2.equalizeHist(curr_frame)

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            prev_frame = cv2.filter2D(prev_frame_equ, -1, kernel)
            curr_frame = cv2.filter2D(curr_frame_equ, -1, kernel)

            #prev_frame = cv2.medianBlur(prev_frame, 5)
            #curr_frame = cv2.medianBlur(curr_frame, 5)

            mag, ang, u, v = of_dense_instance.getFlow(prev_frame, curr_frame)
            '''
            ## filtering
            filtered_orig_mag_np = np.copy(mag)
            data_std = np.std(filtered_orig_mag_np)
            data_mean = np.mean(filtered_orig_mag_np)
            anomaly_cut_off = data_std * 1
            lower_limit = data_mean - anomaly_cut_off
            upper_limit = data_mean + anomaly_cut_off
            filtered_orig_mag_np[filtered_orig_mag_np < lower_limit] = 0
            filtered_orig_mag_np[filtered_orig_mag_np > upper_limit] = 0

            filtered_orig_angles_np = np.copy(ang)
            data_std = np.std(filtered_orig_angles_np)
            data_mean = np.mean(filtered_orig_angles_np)
            anomaly_cut_off = data_std * 1
            lower_limit = data_mean - anomaly_cut_off
            upper_limit = data_mean + anomaly_cut_off
            filtered_orig_angles_np[filtered_orig_angles_np < lower_limit] = 0
            filtered_orig_angles_np[filtered_orig_angles_np > upper_limit] = 0
            '''

            #plt.scatter(ang.flatten(), mag.flatten(), s=1)
            #plt.scatter(filtered_orig_angles_np.flatten(), filtered_orig_mag_np.flatten(), s=1)
            #plt.show()

            orig_mag_l.append(mag)
            orig_angles_l.append(ang)

            orig_u_l.append(u)
            orig_v_l.append(v)


            '''
            b, bins, patches = plt.hist(ang.flatten(), bins=8, range=[0, 360], cumulative=False)
            print("#######################################")
            print(b)
            b_perc_np = (b / sum(b)) * 100.0
            print(b_perc_np)
            
            th_perc = 50.0
            if (((b_perc_np[0] + b_perc_np[7]) > th_perc) or ((b_perc_np[3] + b_perc_np[4]) > th_perc)):
                print("PAN")
                print((b_perc_np[0] + b_perc_np[7]))
                print((b_perc_np[3] + b_perc_np[4]))
                class_name = "PAN"
            elif (((b_perc_np[1] + b_perc_np[2]) > th_perc) or ((b_perc_np[5] + b_perc_np[6]) > th_perc)):
                print("TILT")
                class_name = "TILT"
            else:
                print("NA")
                class_name = "NA"

            '''

            '''
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(median_curr_frame, cmap='gray')
            
            #
            
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.scatter(ang.flatten(), mag.flatten(), s=1)
            #ax0.scatter(filtered_ang, mag.flatten(), s=1)

            #ax1 = fig.add_subplot(gs[0, 1], projection='polar')
            #ax1.scatter(ang.flatten(), mag.flatten(), s=1)
            #plt.ylim(0, 15)
            #plt.xlim(0, 8)
            plt.draw()
            plt.pause(0.1)
            plt.cla()
            '''

            # print("################")
            block_u_mean_l = []
            block_v_mean_l = []

            new_block_mag_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))
            new_block_ang_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))

            curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2RGB)


            block_mag_mean_l = []
            block_ang_mean_l = []
            block_mag_std_l = []
            block_ang_std_l = []
            for r in range(0, self.number_of_blocks):
                for c in range(0, self.number_of_blocks):
                    # block
                    # mag_block = self.getBlock(frame=mag, row=r, col=c)
                    # ang_block = self.getBlock(frame=np.degrees(ang), row=r, col=c)
                    # mag_blocks_per_frame.append(np.mean(mag_block))
                    # ang_blocks_per_frame.append(np.mean(ang_block))
                    u_block, (u_block_center_x, u_block_center_y) = self.getBlock(frame=u, row=r, col=c,
                                                                                  number_of_blocks=self.number_of_blocks)
                    v_block, (v_block_center_x, v_block_center_y) = self.getBlock(frame=v, row=r, col=c,
                                                                                  number_of_blocks=self.number_of_blocks)

                    block_mag, (u_block_center_x, u_block_center_y) = self.getBlock(frame=mag, row=r, col=c,
                                                                                  number_of_blocks=self.number_of_blocks)
                    block_ang, (v_block_center_x, v_block_center_y) = self.getBlock(frame=ang, row=r, col=c,
                                                                                  number_of_blocks=self.number_of_blocks)

                    #block_mag, block_ang = cv2.cartToPolar(u_block, v_block)



                    # mag_mean = int(np.mean(mag_block))
                    # print(mag_mean)

                    block_mag_mean = np.median(block_mag)
                    block_ang_mean = np.median(block_ang)

                    block_mag_std = np.std(block_mag)
                    block_ang_std = np.std(block_ang)

                    new_block_mag_mean_np[r, c] = block_mag_mean
                    new_block_ang_mean_np[r, c] = block_ang_mean

                    block_mag_mean_l.append(block_mag_mean)
                    block_ang_mean_l.append(block_ang_mean)

                    block_mag_std_l.append(block_mag_std)
                    block_ang_std_l.append(block_ang_std)

                    #filtered_u, _ = self.outlier_filter(data_np=u_block.flatten(), alpha=3)
                    #filtered_v, _ = self.outlier_filter(data_np=v_block.flatten(), alpha=3)
                    u_mean = np.median(u_block)
                    v_mean = np.median(v_block)
                    block_u_mean_l.append(u_mean)
                    block_v_mean_l.append(v_mean)


                    cv2.circle(curr_frame_rgb, center=(u_block_center_x, u_block_center_y), radius=1, thickness=1,
                               color=(0, 255, 0))
                    cv2.line(curr_frame_rgb, pt1=(u_block_center_x, u_block_center_y),
                             pt2=(int(u_block_center_x + u_mean), int(u_block_center_y + v_mean)),
                             thickness=1,
                             color=(0, 0, 255))
                    '''
                    
                    print("frame_id: (" + str(i - 1) + "/" 
                            + str(i) 
                            + ") - block " 
                            + str(r) + "-" 
                            + str(c) + ": " 
                            + str(mag_block.shape) 
                            + " - " + str(np.mean(mag_block)) 
                            + " - " + str(np.std(mag_block)) 
                            + " - " + str(np.mean(ang_block)) 
                            + " - " + str(np.std(ang_block)))      
                    '''
                    #cv2.imshow("block " + str(r) + "-" + str(c), mag_block)




            #cv2.imshow("MVF", curr_frame_rgb)
            #k = cv2.waitKey(10)
            if (self.config_instance.save_debug_pkg_flag == True):
                self.video_writer.write(curr_frame_rgb)

            '''
            plt.plot(np.arange(len(block_mag_mean_np)), block_mag_mean_np)
            #plt.plot(np.arange(len(block_v_mean_np)), block_v_mean_np)
            plt.ylim(-70, 70)
            #plt.plot(np.arange(len(filtered_block_mag_mean_np1)), filtered_block_mag_mean_np1)
            #plt.plot(np.arange(len(filtered_block_mag_mean_np2)), filtered_block_mag_mean_np2)
            #plt.plot(np.arange(len(filtered_block_mag_mean_np3)), filtered_block_mag_mean_np3)
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            

            # plot number of features
            #plt.figure(1)

            axs[0].plot(np.arange(len(block_mag_mean_np)), block_mag_mean_np)
            #axs[0].plot(np.arange(len(x_filtered_mag_np)), x_filtered_mag_np)
            axs[1].plot(np.arange(len(block_ang_mean_np)), block_ang_mean_np)
            #axs[1].plot(np.arange(len(x_filtered_ang_np)), x_filtered_ang_np)
            axs[1].set_ylim([-400, 400])
            #axs[2].plot(np.arange(len(u_np)), u_np)
            #axs[2].plot(np.arange(len(filtered_u_np)), filtered_u_np)
            #axs[3].plot(np.arange(len(v_np)), v_np)
            #axs[3].plot(np.arange(len(filtered_v_np)), filtered_v_np)

            plt.grid(True)
            #plt.show()
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            '''

            '''
            #cv2.imshow("orig", curr_frame)
            #k = cv2.waitKey(10)
            plt.hist2d(block_ang_mean_np, block_mag_mean_np, bins=[200, 200], range=((0, 360), (0, 50)))
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            '''


            block_mag_std_np = np.array(block_mag_std_l)
            block_ang_std_np = np.array(block_ang_std_l)

            block_mag_mean_np = np.array(block_mag_mean_l)
            block_ang_mean_np = np.array(block_ang_mean_l)

            frame_mag_mean_np = np.mean(block_mag_mean_np)
            frame_mag_std_np = np.std(block_mag_mean_np)
            frame_ang_mean_np = np.mean(block_ang_mean_np)
            frame_ang_std_np = np.std(block_ang_mean_np)

            all_frame_mag_mean_l.append(frame_mag_mean_np)
            all_frame_ang_mean_l.append(frame_ang_mean_np)


            block_u_mean_np = np.array(block_u_mean_l)
            block_v_mean_np = np.array(block_v_mean_l)

            frame_u_mean_np = np.mean(block_u_mean_np)
            frame_u_std_np = np.std(block_u_mean_np)
            frame_v_mean_np = np.mean(block_v_mean_np)
            frame_v_std_np = np.std(block_v_mean_np)

            all_frame_u_mean_l.append(frame_u_mean_np)
            all_frame_v_mean_l.append(frame_v_mean_np)

            # mag_blocks_per_frame_np = np.array(mag_blocks_per_frame)
            # ang_blocks_per_frame_np = np.array(ang_blocks_per_frame)
            # print(mag_blocks_per_frame_np.shape)
            # print(ang_blocks_per_frame_np)

            # print(ang_blocks_per_frame_np)
            # mag_l_n.append(mag_blocks_per_frame)
            # angles_l_n.append(ang_blocks_per_frame)

            # b, bins, patches = plt.hist(ang_blocks_per_frame_np.flatten(), bins=8, range=[0, 360],
            #                        cumulative=False)  # bins=None, range=None
            # print(b)

            '''
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2RGB)
            dense_flow = cv2.addWeighted(curr_frame, 0.7, rgb, 2, 0)
            cv2.imshow('orig', curr_frame)
            cv2.imshow('flow',rgb)
            cv2.imshow('dense flow', dense_flow)
            k = cv2.waitKey(3) & 0xff
            '''

        orig_mag_l_np = np.array(orig_mag_l)
        orig_angles_l_np = np.array(orig_angles_l)

        print(orig_mag_l_np.shape)
        print(orig_angles_l_np.shape)
        #exit()


        if (self.config_instance.save_debug_pkg_flag == True):
            self.video_writer.release()

        #######################
        MODE = False
        if(MODE == True):
            orig_mag_np = np.array(orig_mag_l)
            orig_angles_np = np.array(orig_angles_l)
            orig_u_np = np.array(orig_u_l)
            orig_v_np = np.array(orig_v_l)

            all_frame_mag_mean_np = np.array(all_frame_mag_mean_l)
            all_frame_ang_mean_np = np.array(all_frame_ang_mean_l)

            #filtered_u_np, _ = self.outlier_filter(orig_u_np.flatten(), alpha=1)
            #filtered_v_np, _ = self.outlier_filter(orig_v_np.flatten(), alpha=1)

            print(np.max(orig_u_np.flatten()))
            print(np.max(orig_v_np.flatten()))

            if(np.abs(np.max(orig_u_np.flatten())) >= np.abs(np.max(orig_v_np.flatten()))):
                number_of_bins = np.abs(np.max(orig_u_np.flatten())) + 1

            if(np.abs(np.max(orig_v_np.flatten())) >= np.abs(np.max(orig_u_np.flatten()))):
                number_of_bins = np.abs(np.max(orig_v_np.flatten())) + 1

            print(number_of_bins)
            print(np.max(orig_mag_np.flatten()))

            orig_mag_mean = np.mean(orig_mag_np.flatten())
            orig_mag_std = np.std(orig_mag_np.flatten())
            orig_angles_mean = np.mean(orig_angles_np.flatten())
            orig_angles_std = np.std(orig_angles_np.flatten())
            print(orig_mag_mean)
            print(orig_mag_std)
            print(orig_angles_mean)
            print(orig_angles_std)

            #plt.hist2d(orig_angles_np.flatten(), orig_mag_np.flatten(), bins=[100, 100], range=((0, 7), (0, 5)))
            #plt.show()
            #exit()

            '''
            # Generate random data:
            N = 10000
            r = .5 + np.random.normal(size=N, scale=.1)
            theta = np.pi / 2 + np.random.normal(size=N, scale=.1)
            print(r.shape)
            print(theta.shape)
            print(orig_mag_np.flatten()[:512].shape)
            '''

            #print(orig_mag_np.shape)
            print(orig_mag_np.flatten().shape)
            #print(orig_angles_np.shape)
            print(orig_angles_np.flatten().shape)


            # Histogramming
            nr = int(number_of_bins)
            ntheta = int(number_of_bins)
            r_edges = np.linspace(0, np.max(orig_mag_np.flatten()), nr + 1)
            theta_edges = np.linspace(0, 2 * np.pi, ntheta + 1)
            #print(theta_edges)
            #print(theta_edges.shape)
            #print(r_edges)
            #print(r_edges.shape)

            '''
            threshold = 0.5
            H, _, _ = np.histogram2d(orig_mag_np.flatten(), np.degrees(orig_angles_np.flatten()), [r_edges, theta_edges], normed=False)
            print(np.max(H))
            print(np.min(H))
            #print(H)
    
            H_normed = (H / np.max(H)) * 1.0
            print(np.max(H_normed))
            print(np.min(H_normed))
            #print(H_normed)
            print(H_normed[H_normed>threshold])
    
            H_normed[H_normed >= threshold] = 1
            H_normed[H_normed < threshold] = 0
    
            
            # Plot
            ax = plt.subplot(111, polar=True)
            Theta, R = np.meshgrid(theta_edges, r_edges)
            ax.pcolormesh(Theta, R, H)
            plt.show()
    
            #print(np.max(filtered_u_np))
            #print(np.max(filtered_v_np))
            
            exit()
            '''

            #orig_angles_flattened_np = orig_angles_np.flatten()
            #orig_mag_flattened_np = orig_mag_np.flatten()
            orig_mag_flattened_np = all_frame_mag_mean_np.flatten()
            orig_angles_flattened_np = all_frame_ang_mean_np.flatten()
            print(orig_mag_flattened_np.shape)
            print(orig_angles_flattened_np.shape)

            #exit()

            # conditions for different movement types
            tilt_up_condition = np.logical_and((orig_angles_flattened_np > 45), (orig_angles_flattened_np < 135))
            tilt_down_condition = np.logical_and((orig_angles_flattened_np > 225), (orig_angles_flattened_np < 315))
            pan_left_condition = np.logical_and((orig_angles_flattened_np > 135), (orig_angles_flattened_np < 225))
            pan_right_condition = np.logical_or(np.logical_and((orig_angles_flattened_np >= 0), (orig_angles_flattened_np < 45)),
                                                np.logical_and((orig_angles_flattened_np > 315), (orig_angles_flattened_np <= 0))
                                                )

            ang_tilt_down = orig_angles_flattened_np[tilt_down_condition]
            mag_tilt_down = orig_mag_flattened_np[tilt_down_condition]
            ang_tilt_up = orig_angles_flattened_np[tilt_up_condition]
            mag_tilt_up = orig_mag_flattened_np[tilt_up_condition]

            ang_pan_left = orig_angles_flattened_np[pan_left_condition]
            mag_pan_left = orig_mag_flattened_np[pan_left_condition]
            ang_pan_right = orig_angles_flattened_np[pan_right_condition]
            mag_pan_right = orig_mag_flattened_np[pan_right_condition]

            mag_tilt_down_mean = np.mean(mag_tilt_down)
            mag_tilt_up_mean = np.mean(mag_tilt_up)
            mag_pan_left_mean = np.mean(mag_pan_left)
            mag_pan_right_mean = np.mean(mag_pan_right)

            data = np.array([mag_tilt_down_mean, mag_tilt_up_mean, mag_pan_left_mean, mag_pan_right_mean])
            data_norm = data / np.max(data)
            print(data_norm)
            mag_tilt_down_mean_norm = data_norm[0]
            mag_tilt_up_mean_norm = data_norm[1]
            mag_pan_left_mean_norm = data_norm[2]
            mag_pan_right_mean_norm = data_norm[3]


            print(f'mean angle of tilt_downs: {np.mean(ang_tilt_down)}')
            print(f'mean mag of tilt_downs: {np.mean(mag_tilt_down)}')

            print(f'mean angle of tilt_ups: {np.mean(ang_tilt_up)}')
            print(f'mean mag of tilt_ups: {np.mean(mag_tilt_up)}')

            print(f'mean angle of pan_left: {np.mean(ang_pan_left)}')
            print(f'mean mag of pan_left: {np.mean(mag_pan_left)}')

            print(f'mean angle of pan_right: {np.mean(ang_pan_right)}')
            print(f'mean mag of pan_right: {np.mean(mag_pan_right)}')


            #print(orig_mag_np.shape)
            #print(orig_angles_np.shape)
            #print(orig_angles_np.flatten().shape)
            #plt.hist(orig_angles_np.flatten())
            b, bins, patches = plt.hist(orig_angles_flattened_np, bins=8, range=[0, 360],
                                        cumulative=False)  # bins=None, range=None
            #plt.show()
            #print(b)
            #print((b / sum(b)) * 100.0)
            #print(sum((b / sum(b)) * 100.0))
            #print(bins)

            b_perc_np = (b / sum(b)) * 100.0
            #print(sum(b_perc_np))

            # ( sum 1 8  and sum 4 5 ) > 50 % --> PAN
            # ( sum 2 3  and sum 6 7 ) > 50 % --> TILT
            # otherwise NA

            mag_tilt_down_mean_norm = data_norm[0]
            mag_tilt_up_mean_norm = data_norm[1]
            mag_pan_left_mean_norm = data_norm[2]
            mag_pan_right_mean_norm = data_norm[3]
            print(f'mag TILT DOWN normed: {mag_tilt_down_mean_norm}')
            print(f'mag TILT UP normed: {mag_tilt_up_mean_norm}')
            print(f'mag PAN LEFT normed: {mag_pan_left_mean_norm}')
            print(f'mag PAN RIGHT normed: {mag_pan_right_mean_norm}')

            print(f'ang TILT DOWN: {(b_perc_np[5] + b_perc_np[6])}')
            print(f'ang TILT UP: {(b_perc_np[1] + b_perc_np[2])}')
            print(f'ang PAN LEFT: {(b_perc_np[0] + b_perc_np[7])}')
            print(f'ang PAN RIGHT: {(b_perc_np[3] + b_perc_np[4])}')

            data_new = [(b_perc_np[5] + b_perc_np[6])*mag_tilt_down_mean_norm,
                     (b_perc_np[1] + b_perc_np[2])*mag_tilt_up_mean_norm,
                     (b_perc_np[0] + b_perc_np[7])*mag_pan_left_mean_norm,
                     (b_perc_np[3] + b_perc_np[4])*mag_pan_right_mean_norm]

            data_mean = np.mean([(b_perc_np[5] + b_perc_np[6])*mag_tilt_down_mean_norm,
                     (b_perc_np[1] + b_perc_np[2])*mag_tilt_up_mean_norm,
                     (b_perc_np[0] + b_perc_np[7])*mag_pan_left_mean_norm,
                     (b_perc_np[3] + b_perc_np[4])*mag_pan_right_mean_norm])
            data_std = np.std([(b_perc_np[5] + b_perc_np[6]) * mag_tilt_down_mean_norm,
                                 (b_perc_np[1] + b_perc_np[2]) * mag_tilt_up_mean_norm,
                                 (b_perc_np[0] + b_perc_np[7]) * mag_pan_left_mean_norm,
                                 (b_perc_np[3] + b_perc_np[4]) * mag_pan_right_mean_norm])

            print(data_mean)
            print(data_std)

            stat_condition = abs(data_mean - data_std) > data_std


            # first level classification - ANGLES
            th_perc = 50.0
            if (((b_perc_np[0] + b_perc_np[7]) > th_perc) or ((b_perc_np[3] + b_perc_np[4]) > th_perc) ):
                print("PAN")
                class_name = "PAN"
            elif(((b_perc_np[1] + b_perc_np[2]) > th_perc) or ((b_perc_np[5] + b_perc_np[6]) > th_perc)):
                print("TILT")
                class_name = "TILT"
            else:
                print("NA")
                class_name = "NA"

            # second level classification - MAG
            print("['PAN', 'TILT', 'TILT', 'PAN', 'PAN', 'TILT', 'TILT', 'PAN']")
            print(b)
            print(np.round(b_perc_np, 3))

            #exit()
            return class_name

        #######################

        mag_np = np.array(all_frame_mag_mean_l)
        ang_np = np.array(all_frame_ang_mean_l)
        u_np = np.array(all_frame_u_mean_l)
        v_np = np.array(all_frame_v_mean_l)

        filtered_u_np = self.window_filter(u_np, window_size=20)
        filtered_v_np = self.window_filter(v_np, window_size=20)

        # window filtering
        #filtered_mag_np = self.window_filter(mag_np, window_size=10)
        #filtered_ang_np = self.window_filter(ang_np, window_size=10)

        new_mag_np = np.array(new_blocks_mag_l)
        new_ang_np = np.array(new_blocks_ang_l)

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude

        #mag_condition_pan = abs(x_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)
        #mag_condition_tilt = abs(y_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)

        print("AAA")
        print(mag_np.shape)
        #x_filtered_mag_np = filtered_mag_np[filtered_mag_np > THRESHOLD1]
        x_filtered_mag_np = self.window_filter(mag_np, window_size=10) #[mag_np > th]
        x_filtered_ang_np = self.window_filter(ang_np, window_size=10) #[mag_np > th]
        #x_filtered_ang_np = ang_np[filtered_mag_np > THRESHOLD1]

        print(np.mean(x_filtered_mag_np))
        print(np.mean(x_filtered_ang_np))


        '''
        plt.polar(np.radians(ang_np), mag_np, 'g.')
        plt.polar(np.radians(x_filtered_ang_np), x_filtered_mag_np, 'r.')
        plt.show()
        
        exit()
        
       
        plt.plot(x_filtered_ang_np)
        plt.show()
        
        b, bins, patches = plt.hist(ang_np, bins=8, range=[0, 360],
                                    cumulative=False)  # bins=None, range=None
        b, bins, patches = plt.hist(x_filtered_ang_np, bins=8, range=[0, 360],
                                    cumulative=False)  # bins=None, range=None
        plt.show()
        
        plt.hist2d(mag_np, ang_np)
        plt.show()
        '''

        '''
        #fig, axs = plt.subplots(4)
        for r in range(0, 32):
            for c in range(0, 32):
                # TODO: outlier detection Mag per block over all frames
                tmp_mag_frame_mean = new_mag_np[:, r, c]
                tmp_ang_frame_mean = new_ang_np[:, r, c]

                # filtering mag
                filtered_mag_frame_per_block = self.window_filter(tmp_mag_frame_mean, window_size=10)

                relevant_angles = np.copy(tmp_ang_frame_mean)
                relevant_angles[filtered_mag_frame_per_block > THRESHOLD1] = tmp_ang_frame_mean[
                    filtered_mag_frame_per_block > THRESHOLD1]
                #relevant_angles[filtered_mag_frame_per_block > THRESHOLD] = tmp_ang_frame_mean[
                #    filtered_mag_frame_per_block > THRESHOLD]
                relevant_angles[filtered_mag_frame_per_block <= THRESHOLD1] = 0
                relevant_angles[filtered_mag_frame_per_block > THRESHOLD2] = 0
                #n_filtered_mag_np = self.window_filter(new_mag_np[:, a, a], window_size=10)

                #axs[0].plot(np.arange(len(filtered_mag_np)), filtered_mag_np)
                #axs[1].plot(np.arange(len(filtered_mag_frame_per_block)), filtered_mag_frame_per_block)
                #axs[2].plot(np.arange(len(relevant_angles)), relevant_angles)
                plt.polar(tmp_mag_frame_mean, tmp_ang_frame_mean, 'ro')


                #exit()

            #n_filtered_ang_np = self.window_filter(new_ang_np[:, a, a], window_size=10)

            #axs[0].plot(np.arange(len(new_ang_np)), new_ang_np[:, a, a])
            #axs[0].plot(np.arange(len(n_filtered_ang_np)), n_filtered_ang_np)
            #axs[1].plot(np.arange(len(n_filtered_mag_np)), n_filtered_mag_np)

        plt.show()
        exit()
        '''

        '''
        # plot number of features
        #plt.figure(1)
        fig, axs = plt.subplots(4)
        axs[0].plot(np.arange(len(mag_np)), mag_np)
        axs[0].plot(np.arange(len(x_filtered_mag_np)), x_filtered_mag_np)
        axs[1].plot(np.arange(len(ang_np)), ang_np)
        axs[1].plot(np.arange(len(x_filtered_ang_np)), x_filtered_ang_np)
        axs[1].set_ylim([-400, 400])
        axs[2].plot(np.arange(len(u_np)), u_np)
        axs[2].plot(np.arange(len(filtered_u_np)), filtered_u_np)
        axs[3].plot(np.arange(len(v_np)), v_np)
        axs[3].plot(np.arange(len(filtered_v_np)), filtered_v_np)

        plt.grid(True)
        plt.show()
        #plt.draw()
        #plt.pause(0.02)
        
        

        print("predicted class_name: " + str(class_name))
        
        exit()
        '''

        return x_filtered_mag_np, x_filtered_ang_np, filtered_u_np, filtered_v_np

    def runDense_NEW(self):
        frames_np = np.squeeze(self.video_frames)

        of_dense_instance = OpticalFlow_Dense()
        hsv = np.zeros_like(frames_np[0])
        hsv[..., 1] = 255

        angles_l_n = []
        mag_l_n = []

        all_frames_blocks_u_l = []
        all_frames_blocks_v_l = []
        all_frames_blocks_mag_l = []
        all_frames_blocks_ang_l = []

        all_frame_u_mean_l = []
        all_frame_v_mean_l = []
        all_frame_mag_mean_l = []
        all_frame_ang_mean_l = []
        step_size = 1

        # fig = plt.figure()
        # gs = gridspec.GridSpec(nrows=1, ncols=2)

        orig_mag_l = []
        orig_angles_l = []
        orig_u_l = []
        orig_v_l = []

        if (self.config_instance.save_debug_pkg_flag == True):
            frame_size = (frames_np[0].shape[1], frames_np[0].shape[0])
            self.video_writer = cv2.VideoWriter("/data/share/dense_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12,
                                                frame_size)

        for i in range(step_size, len(frames_np), step_size):
            prev_frame = frames_np[i - step_size]
            curr_frame = frames_np[i]

            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            #prev_frame_equ = cv2.equalizeHist(prev_frame)
            #curr_frame_equ = cv2.equalizeHist(curr_frame)

            #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            #prev_frame = cv2.filter2D(prev_frame, -1, kernel)
            #curr_frame = cv2.filter2D(curr_frame, -1, kernel)

            #prev_frame = cv2.boxFilter(prev_frame, -1, (5, 5))
            #curr_frame = cv2.boxFilter(curr_frame, -1, (5, 5))


            #prev_frame = cv2.bilateralFilter(prev_frame, 9, 75, 75)
            #curr_frame = cv2.bilateralFilter(curr_frame, 9, 75, 75)

            #prev_frame = cv2.GaussianBlur(prev_frame,(15,15),0)
            #curr_frame =  cv2.GaussianBlur(curr_frame,(15,15),0)

            mag, ang, u, v = of_dense_instance.getFlow(prev_frame, curr_frame)

            '''
            hsv = np.zeros((curr_frame.shape[0], curr_frame.shape[1], 3), dtype=np.uint8)
            print(hsv.shape)
            print(curr_frame.shape)
            hsv[..., 1] = 255

            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("colored flow", bgr)
            cv2.imshow("orig", curr_frame)
            cv2.waitKey(10)
            '''

            # plt.scatter(ang.flatten(), mag.flatten(), s=1)
            # plt.scatter(filtered_orig_angles_np.flatten(), filtered_orig_mag_np.flatten(), s=1)
            # plt.show()

            orig_mag_l.append(mag)
            orig_angles_l.append(ang)

            orig_u_l.append(u)
            orig_v_l.append(v)

            # print("################")


            new_block_mag_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))
            new_block_ang_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))

            new_block_u_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))
            new_block_v_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))

            curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2RGB)

            block_mag_mean_l = []
            block_ang_mean_l = []
            block_mag_std_l = []
            block_ang_std_l = []

            block_u_mean_l = []
            block_v_mean_l = []

            all_u_block_center_x = []
            all_u_block_center_y = []

            for r in range(0, self.number_of_blocks):
                for c in range(0, self.number_of_blocks):
                    # block
                    # mag_block = self.getBlock(frame=mag, row=r, col=c)
                    # ang_block = self.getBlock(frame=np.degrees(ang), row=r, col=c)
                    # mag_blocks_per_frame.append(np.mean(mag_block))
                    # ang_blocks_per_frame.append(np.mean(ang_block))
                    u_block, (u_block_center_x, u_block_center_y) = self.getBlock(frame=u, row=r, col=c,
                                                                                  number_of_blocks=self.number_of_blocks)
                    v_block, (v_block_center_x, v_block_center_y) = self.getBlock(frame=v, row=r, col=c,
                                                                                  number_of_blocks=self.number_of_blocks)

                    block_mag, (u_block_center_x, u_block_center_y) = self.getBlock(frame=mag, row=r, col=c,
                                                                                    number_of_blocks=self.number_of_blocks)
                    block_ang, (v_block_center_x, v_block_center_y) = self.getBlock(frame=ang, row=r, col=c,
                                                                                    number_of_blocks=self.number_of_blocks)

                    all_u_block_center_x.append(u_block_center_x)
                    all_u_block_center_y.append(u_block_center_y)

                    # block_mag, block_ang = cv2.cartToPolar(u_block, v_block)

                    # mag_mean = int(np.mean(mag_block))
                    # print(mag_mean)

                    block_mag_mean = np.median(block_mag)
                    block_ang_mean = np.median(block_ang)
                    block_mag_std = np.std(block_mag)
                    block_ang_std = np.std(block_ang)

                    new_block_mag_mean_np[r, c] = block_mag_mean
                    new_block_ang_mean_np[r, c] = block_ang_mean



                    block_mag_mean_l.append(block_mag_mean)
                    block_ang_mean_l.append(block_ang_mean)

                    block_mag_std_l.append(block_mag_std)
                    block_ang_std_l.append(block_ang_std)

                    # filtered_u, _ = self.outlier_filter(data_np=u_block.flatten(), alpha=3)
                    # filtered_v, _ = self.outlier_filter(data_np=v_block.flatten(), alpha=3)
                    block_u_mean = np.median(u_block)
                    block_v_mean = np.median(v_block)
                    block_u_mean_l.append(block_u_mean)
                    block_v_mean_l.append(block_v_mean)

                    new_block_u_mean_np[r, c] = block_u_mean
                    new_block_v_mean_np[r, c] = block_v_mean

                    '''
                    cv2.circle(curr_frame_rgb, center=(u_block_center_x, u_block_center_y), radius=1, thickness=1,
                               color=(0, 255, 0))
                    cv2.line(curr_frame_rgb, pt1=(u_block_center_x, u_block_center_y),
                             pt2=(int(u_block_center_x + u_mean), int(u_block_center_y + v_mean)),
                             thickness=1,
                             color=(0, 0, 255))
                    

                    print("frame_id: (" + str(i - 1) + "/" 
                            + str(i) 
                            + ") - block " 
                            + str(r) + "-" 
                            + str(c) + ": " 
                            + str(mag_block.shape) 
                            + " - " + str(np.mean(mag_block)) 
                            + " - " + str(np.std(mag_block)) 
                            + " - " + str(np.mean(ang_block)) 
                            + " - " + str(np.std(ang_block)))      
                    '''
                    # cv2.imshow("block " + str(r) + "-" + str(c), mag_block)

            # cv2.imshow("MVF", curr_frame_rgb)
            # k = cv2.waitKey(10)
            if (self.config_instance.save_debug_pkg_flag == True):
                self.video_writer.write(curr_frame_rgb)

            '''
          
            # plot number of features
            #plt.figure(1)

            axs[0].plot(np.arange(len(block_mag_mean_np)), block_mag_mean_np)
            #axs[0].plot(np.arange(len(x_filtered_mag_np)), x_filtered_mag_np)
            axs[1].plot(np.arange(len(block_ang_mean_np)), block_ang_mean_np)
            #axs[1].plot(np.arange(len(x_filtered_ang_np)), x_filtered_ang_np)
            axs[1].set_ylim([-400, 400])
            #axs[2].plot(np.arange(len(u_np)), u_np)
            #axs[2].plot(np.arange(len(filtered_u_np)), filtered_u_np)
            #axs[3].plot(np.arange(len(v_np)), v_np)
            #axs[3].plot(np.arange(len(filtered_v_np)), filtered_v_np)

            plt.grid(True)
            #plt.show()
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            '''

            '''
            #cv2.imshow("orig", curr_frame)
            #k = cv2.waitKey(10)
            plt.hist2d(block_ang_mean_np, block_mag_mean_np, bins=[200, 200], range=((0, 360), (0, 50)))
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            '''

            block_mag_mean_np = np.array(block_mag_mean_l)
            block_ang_mean_np = np.array(block_ang_mean_l)

            block_u_mean_np = np.array(block_u_mean_l)
            block_v_mean_np = np.array(block_v_mean_l)


            #print(block_u_mean_np)
            #print(block_u_mean_np.shape)
            #print(block_v_mean_np)
            #print(block_v_mean_np.shape)

            #print(new_block_u_mean_np)
            #print(new_block_u_mean_np.shape)
            #print(new_block_v_mean_np)
            #print(new_block_v_mean_np.shape)

            all_frames_blocks_u_l.append(new_block_u_mean_np)
            all_frames_blocks_v_l.append(new_block_v_mean_np)

            all_frames_blocks_mag_l.append(new_block_mag_mean_np)
            all_frames_blocks_ang_l.append(new_block_ang_mean_np)


            '''
            frame_mag_mean_np = np.mean(block_mag_mean_np)
            frame_ang_mean_np = np.mean(block_ang_mean_np)

            all_frame_mag_mean_l.append(frame_mag_mean_np)
            all_frame_ang_mean_l.append(frame_ang_mean_np)



            frame_u_mean_np = np.mean(block_u_mean_np)
            frame_u_std_np = np.std(block_u_mean_np)
            frame_v_mean_np = np.mean(block_v_mean_np)
            frame_v_std_np = np.std(block_v_mean_np)

            all_frame_u_mean_l.append(frame_u_mean_np)
            all_frame_v_mean_l.append(frame_v_mean_np)

            # mag_blocks_per_frame_np = np.array(mag_blocks_per_frame)
            # ang_blocks_per_frame_np = np.array(ang_blocks_per_frame)
            # print(mag_blocks_per_frame_np.shape)
            # print(ang_blocks_per_frame_np)

            # print(ang_blocks_per_frame_np)
            # mag_l_n.append(mag_blocks_per_frame)
            # angles_l_n.append(ang_blocks_per_frame)

            # b, bins, patches = plt.hist(ang_blocks_per_frame_np.flatten(), bins=8, range=[0, 360],
            #                        cumulative=False)  # bins=None, range=None
            # print(b)
            '''


            '''
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2RGB)
            dense_flow = cv2.addWeighted(curr_frame, 0.7, rgb, 2, 0)
            cv2.imshow('orig', curr_frame)
            cv2.imshow('flow',rgb)
            cv2.imshow('dense flow', dense_flow)
            k = cv2.waitKey(3) & 0xff
            '''

        all_frames_blocks_mag_np = np.array(all_frames_blocks_mag_l)
        all_frames_blocks_ang_np = np.array(all_frames_blocks_ang_l)
        all_frames_blocks_u_np = np.array(all_frames_blocks_u_l)
        all_frames_blocks_v_np = np.array(all_frames_blocks_v_l)
        print(all_frames_blocks_u_np.shape)
        print(all_frames_blocks_v_np.shape)

        all_delta_u = []
        all_delta_v = []

        for i in range(0, len(all_frames_blocks_u_np) - 1):
            u_curr = all_frames_blocks_u_np[i]
            u_next = all_frames_blocks_u_np[i+1]
            v_curr = all_frames_blocks_v_np[i]
            v_next = all_frames_blocks_v_np[i+1]
            delta_u = u_next - u_curr
            delta_v = v_next - v_curr
            all_delta_u.append(delta_u)
            all_delta_v.append(delta_v)

        all_delta_u_np = np.array(all_delta_u)
        all_delta_v_np = np.array(all_delta_v)
        print(all_delta_u_np.shape)
        print(all_delta_v_np.shape)

        # check significance


        # check consistency
        k = 10
        n = 15

        for i in range(0, len(frames_np) - k - 1):
            sum_delta_u = all_delta_u[i + 0]
            sum_delta_v = all_delta_v[i + 0]
            for j in range(1, k):
                sum_delta_u = sum_delta_u + all_delta_u[i + j]
                sum_delta_v = sum_delta_v + all_delta_v[i + j]
            mu_delta_u = (1/k) * sum_delta_u
            mu_delta_v = (1/k) * sum_delta_v
            print(mu_delta_u)
            print(mu_delta_u.shape)

            sum_sigma_delta_u = np.square(all_frames_blocks_u_np[i + 0] - mu_delta_u)
            sum_sigma_delta_v = np.square(all_frames_blocks_v_np[i + 0] - mu_delta_v)
            for j in range(1, k):
                sum_sigma_delta_u = sum_sigma_delta_u + np.square(all_frames_blocks_u_np[i + j] - mu_delta_u)
                sum_sigma_delta_v = sum_sigma_delta_v + np.square(all_frames_blocks_v_np[i + j] - mu_delta_v)
            sigma_delta_u = np.sqrt(1 / (k - 1) * sum_sigma_delta_u)
            sigma_delta_v = np.sqrt(1 / (k - 1) * sum_sigma_delta_v)
            print(sigma_delta_u)
            print(sigma_delta_u.shape)

            tmp_sum = np.sqrt((np.square(all_frames_blocks_u_np[i + 0]) + np.square(all_frames_blocks_v_np[i + 0])))
            for j in range(1, k):
                tmp_sum = tmp_sum + np.sqrt((np.square(all_frames_blocks_u_np[i + j]) + np.square(all_frames_blocks_v_np[i + j])))
            mu_dist = 1/k * tmp_sum

            #print(mu_dist)
            #print(mu_dist.shape)

            t1 = 0.3
            t2 = 6

            print(np.min(mu_dist))
            print(np.max(mu_dist))

            print(np.min(np.sqrt(np.square(sigma_delta_u) + np.square(sigma_delta_v)) ))
            print(np.max(np.sqrt(np.square(sigma_delta_u) + np.square(sigma_delta_v)) ))

            condition1 = mu_dist > t1
            condition2 = np.sqrt(np.square(sigma_delta_u) + np.square(sigma_delta_v)) < t2
            final_mask = np.logical_and(condition1, condition2)
            #print(condition1)
            #print(condition2)
            #print(final_mask)

            print(np.unique(final_mask, return_counts=True))
            print(np.unique(condition2, return_counts=True))
            print(np.unique(condition1, return_counts=True))

            #all_frames_blocks_u_np[i][final_mask == False] = 0
            #all_frames_blocks_v_np[i][final_mask == False] = 0
            #all_frames_blocks_mag_np[i][final_mask == False] = 0
            #all_frames_blocks_ang_np[i][final_mask == False] = 0

            #mv_blocks_u_np[i][final_mask == True] = 0
            #mv_blocks_v_np[i][final_mask == True] = 0
            #mv_mag_np[i][final_mask == True] = 0
            #mv_ang_np[i][final_mask == True] = 0

            tmp_sum_mvi_u = 0
            tmp_sum_mvi_v = 0
            tmp_sum_mv_u = 0
            tmp_sum_mv_v = 0


            for j in range(i, i + n - 1):

                sum_mvi_blocks_u_np = np.sum(np.abs(all_frames_blocks_u_np[i][final_mask == True]))
                sum_mvi_blocks_v_np = np.sum(np.abs(all_frames_blocks_v_np[i][final_mask == True]))
                sum_mv_blocks_u_np = np.sum(np.abs(all_frames_blocks_u_np[i]))
                sum_mv_blocks_v_np = np.sum(np.abs(all_frames_blocks_v_np[i]))

                tmp_sum_mvi_u = tmp_sum_mvi_u + sum_mvi_blocks_u_np
                tmp_sum_mvi_v = tmp_sum_mvi_v + sum_mvi_blocks_v_np

                tmp_sum_mv_u = tmp_sum_mv_u + sum_mv_blocks_u_np
                tmp_sum_mv_v = tmp_sum_mv_v + sum_mv_blocks_v_np

                if(tmp_sum_mvi_u < tmp_sum_mv_u * 0.1) and (tmp_sum_mvi_v < tmp_sum_mv_v * 0.1):
                    print("STATIC")
                else:
                    print("OTHERS")
                    tmp_u = np.expand_dims(all_frames_blocks_u_np[i][final_mask == True], axis=1)
                    tmp_v = np.expand_dims(all_frames_blocks_v_np[i][final_mask == True], axis=1)
                    A = np.concatenate((tmp_u, tmp_v), axis=1)



                    '''
                    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
                    print(u)
                    print(s)
                    print(vh.T)

                    X_svd = np.dot(u, np.diag(s))
                    print(X_svd)

                    exit()
                    '''

                    '''
                    #print(tmp_u.shape)
                    #print(tmp_v.shape)
                    print(A)
                    #plt.figure()
                    plt.scatter(tmp_u, tmp_v)
                    #plt.show()
                    #exit()
                    plt.draw()
                    plt.pause(0.01)
                    plt.cla()
                    '''



                    #exit()
            #print("#######")
            #print(tmp_sum_mvi_u)
            #print(tmp_sum_mvi_v)

            #print(tmp_sum_mv_u)
            #print(tmp_sum_mv_v)

            #print(all_frames_blocks_v_np[i])
            #exit()

            #plt.imshow(frames_np[i])
            #plt.draw()
            #plt.pause(0.01)

            '''
            for s in range(0, len(all_u_block_center_x)):
                cv2.circle(frames_np[i], center=(all_u_block_center_x[s], all_u_block_center_y[s]), radius=1, thickness=1,
                           color=(0, 255, 0))
                cv2.line(frames_np[i], pt1=(all_u_block_center_x[s], all_u_block_center_y[s]),
                         pt2=(int(all_u_block_center_x[s] + all_frames_blocks_u_np[i].flatten()[s]), int(all_u_block_center_y[s] + all_frames_blocks_v_np[i].flatten()[s])),
                         thickness=1,
                         color=(0, 0, 255))

            cv2.imshow("orig", frames_np[i])
            s = cv2.waitKey(10)
            '''

            '''
            #plt.figure()
            plt.polar(np.radians(all_frames_blocks_ang_np[i].flatten()), all_frames_blocks_mag_np[i].flatten(), 'g.')
            plt.draw()
            plt.pause(0.01)
            #plt.cla()
            '''

        print(all_frames_blocks_u_np.shape) # Nx32x32
        print(all_frames_blocks_v_np.shape)
        n = 5


        x_l = []
        y_l = []
        for i in range(0, len(all_frames_blocks_u_np) - n):
            tmp_sum_mvi_u = 0
            tmp_sum_mv_u = 0
            tmp_sum_mvi_v = 0
            tmp_sum_mv_v = 0

            for j in range(i, i+n-1):
                tmp_sum_mvi_u = tmp_sum_mvi_u + np.abs(all_frames_blocks_u_np[i].flatten()[j])
                tmp_sum_mv_u = tmp_sum_mv_u + np.abs(all_frames_blocks_u_np[i].flatten()[j])

                tmp_sum_mvi_v = tmp_sum_mvi_v + np.abs(all_frames_blocks_v_np[i].flatten()[j])
                tmp_sum_mv_v = tmp_sum_mv_v + np.abs(all_frames_blocks_v_np[i].flatten()[j])

            print("#######")
            print(tmp_sum_mvi_u)
            print(tmp_sum_mvi_v)

            print(tmp_sum_mv_u)
            print(tmp_sum_mv_v)

            x_l.append(tmp_sum_mvi_u)
            y_l.append(tmp_sum_mvi_v)

            plt.plot(np.arange(0, len(x_l)), np.array(x_l))
            plt.plot(np.arange(0, len(y_l)), np.array(y_l))
            plt.draw()
            plt.pause(0.01)
            plt.cla()




        exit()


        orig_mag_l_np = np.array(orig_mag_l)
        orig_angles_l_np = np.array(orig_angles_l)

        print(orig_mag_l_np.shape)
        print(orig_angles_l_np.shape)
        # exit()

        if (self.config_instance.save_debug_pkg_flag == True):
            self.video_writer.release()

        #######################

        mag_np = np.array(all_frame_mag_mean_l)
        ang_np = np.array(all_frame_ang_mean_l)
        u_np = np.array(all_frame_u_mean_l)
        v_np = np.array(all_frame_v_mean_l)

        filtered_u_np = self.window_filter(u_np, window_size=20)
        filtered_v_np = self.window_filter(v_np, window_size=20)

        # window filtering
        # filtered_mag_np = self.window_filter(mag_np, window_size=10)
        # filtered_ang_np = self.window_filter(ang_np, window_size=10)

        new_mag_np = np.array(new_blocks_mag_l)
        new_ang_np = np.array(new_blocks_ang_l)

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude

        # mag_condition_pan = abs(x_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)
        # mag_condition_tilt = abs(y_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)

        print("AAA")
        print(mag_np.shape)
        # x_filtered_mag_np = filtered_mag_np[filtered_mag_np > THRESHOLD1]
        x_filtered_mag_np = self.window_filter(mag_np, window_size=10)  # [mag_np > th]
        x_filtered_ang_np = self.window_filter(ang_np, window_size=10)  # [mag_np > th]
        # x_filtered_ang_np = ang_np[filtered_mag_np > THRESHOLD1]

        print(np.mean(x_filtered_mag_np))
        print(np.mean(x_filtered_ang_np))

        return x_filtered_mag_np, x_filtered_ang_np, filtered_u_np, filtered_v_np

    def runDense_v3(self):
        frames_np = np.squeeze(self.video_frames)

        of_dense_instance = OpticalFlow_Dense()
        hsv = np.zeros_like(frames_np[0])
        hsv[..., 1] = 255

        # calcuate optical flow for all frames in the shot
        frm_u_l = []
        frm_v_l = []
        frm_mag_l = []
        frm_ang_l = []
        for i in range(0, len(frames_np)-1):
            curr_frame = frames_np[i]
            nxt_frame = frames_np[i + 1]

            curr_frame = cv2.equalizeHist(curr_frame)
            nxt_frame = cv2.equalizeHist(nxt_frame)

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            curr_frame = cv2.filter2D(curr_frame, -1, kernel)
            nxt_frame = cv2.filter2D(nxt_frame, -1, kernel)

            frm_mag, frm_ang, frm_u, frm_v = of_dense_instance.getFlow(curr_frame, nxt_frame)
            frm_u_l.append(frm_u)
            frm_v_l.append(frm_v)
            frm_mag_l.append(frm_mag)
            frm_ang_l.append(frm_ang)

        frm_u_np = np.array(frm_u_l)
        frm_v_np = np.array(frm_v_l)
        frm_mag_np = np.array(frm_mag_l)
        frm_ang_np = np.array(frm_ang_l)
        print(frm_u_np.shape)
        print(frm_v_np.shape)
        print(frm_mag_np.shape)
        print(frm_ang_np.shape)

        # split into blocks
        all_mb_u_l = []
        all_mb_v_l = []
        all_mb_mag_l = []
        all_mb_ang_l = []
        all_mb_x_l = []
        all_mb_y_l = []
        for i in range(0, len(frm_u_np)):
            frm_u = frm_u_np[i]
            frm_v = frm_v_np[i]
            frm_mag = frm_mag_np[i]
            frm_ang = frm_ang_np[i]

            frm_mb_u_l = []
            frm_mb_v_l = []
            frm_mb_mag_l = []
            frm_mb_ang_l = []
            mb_x_l = []
            mb_y_l = []
            for r in range(0, self.number_of_blocks):
                for c in range(0, self.number_of_blocks):
                    # block
                    mb_u, (mb_center_x, mb_center_y) = self.getBlock(
                        frame=frm_u,
                        row=r,
                        col=c,
                        number_of_blocks=self.number_of_blocks)
                    mb_v, (mb_center_x, mb_center_y) = self.getBlock(
                        frame=frm_v,
                        row=r,
                        col=c,
                        number_of_blocks=self.number_of_blocks)
                    mb_mag, (mb_center_x, mb_center_y) = self.getBlock(
                        frame=frm_mag,
                        row=r,
                        col=c,
                        number_of_blocks=self.number_of_blocks)
                    mb_ang, (mb_center_x, mb_center_y) = self.getBlock(
                        frame=frm_ang,
                        row=r,
                        col=c,
                        number_of_blocks=self.number_of_blocks)

                    frm_mb_u_l.append(np.mean(mb_u))
                    frm_mb_v_l.append(np.mean(mb_v))
                    frm_mb_mag_l.append(np.mean(mb_mag))
                    frm_mb_ang_l.append(np.mean(mb_ang))
                    mb_x_l.append(mb_center_x)
                    mb_y_l.append(mb_center_y)

            all_mb_u_l.append(np.array(frm_mb_u_l))
            all_mb_v_l.append(np.array(frm_mb_v_l))
            all_mb_mag_l.append(np.array(frm_mb_mag_l))
            all_mb_ang_l.append(np.array(frm_mb_ang_l))
            all_mb_x_l.append(np.array(mb_x_l))
            all_mb_y_l.append(np.array(mb_y_l))

        all_mb_u_np = np.array(all_mb_u_l)
        all_mb_v_np = np.array(all_mb_v_l)
        all_mb_mag_np = np.array(all_mb_mag_l)
        all_mb_ang_np = np.array(all_mb_ang_l)
        all_mb_x_np = np.array(all_mb_x_l)
        all_mb_y_np = np.array(all_mb_y_l)
        print(all_mb_u_np.shape)
        print(all_mb_v_np.shape)
        print(all_mb_mag_np.shape)
        print(all_mb_ang_np.shape)
        print(all_mb_x_np.shape)
        print(all_mb_y_np.shape)

        k = 10
        n = 5
        t1 = 1.8
        t2 = 2.9
        mvi_mv_ratio = 0.10

        all_filter_masks_l = []
        all_seq_mb_delta_u_l = []
        all_seq_mb_delta_v_l = []
        last_conditions = np.zeros((self.number_of_blocks*self.number_of_blocks)).astype('bool')
        last_conditions.fill(False)

        for i in range(0, len(all_mb_u_np) - k):
            # calculate delta_u and delta_v - horizontal and vertical displacements
            seq_mb_delta_u = []
            seq_mb_delta_v = []
            for j in range(0, k):
                u_curr = all_mb_u_np[i + j]
                v_curr = all_mb_v_np[i + j]
                u_next = all_mb_u_np[i + j + 1]
                v_next = all_mb_v_np[i + j + 1]

                delta_u = u_next - u_curr
                delta_v = v_next - v_curr
                seq_mb_delta_u.append(delta_u)
                seq_mb_delta_v.append(delta_v)
            seq_mb_delta_u_np = np.array(seq_mb_delta_u)
            seq_mb_delta_v_np = np.array(seq_mb_delta_v)

            print("####### DELTA U-V #########")
            print(i)
            print(seq_mb_delta_u_np)
            print(seq_mb_delta_u_np.shape)
            print(seq_mb_delta_v_np)
            print(seq_mb_delta_v_np.shape)
            #continue

            # check significance and consistency
            seq_mb_mu_delta_u = np.mean(seq_mb_delta_u_np, axis=0)
            seq_mb_mu_delta_v = np.mean(seq_mb_delta_v_np, axis=0)

            all_seq_mb_delta_u_l.append(seq_mb_mu_delta_u)
            all_seq_mb_delta_v_l.append(seq_mb_mu_delta_v)

            sum_seq_mb_sigma_delta_u = np.square(seq_mb_delta_u_np[0] - seq_mb_mu_delta_u)
            sum_seq_mb_sigma_delta_v = np.square(seq_mb_delta_v_np[0] - seq_mb_mu_delta_v)
            for j in range(1, k):
                sum_seq_mb_sigma_delta_u = sum_seq_mb_sigma_delta_u + np.square(seq_mb_delta_u_np[j] - seq_mb_mu_delta_u)
                sum_seq_mb_sigma_delta_v = sum_seq_mb_sigma_delta_v + np.square(seq_mb_delta_v_np[j] - seq_mb_mu_delta_v)
            seq_mb_sigma_delta_u = np.sqrt((1 / (k - 1)) * sum_seq_mb_sigma_delta_u)
            seq_mb_sigma_delta_v = np.sqrt((1 / (k - 1)) * sum_seq_mb_sigma_delta_v)

            print("####### SIGMA U-V #########")
            print(i)
            print(seq_mb_sigma_delta_u)
            print(seq_mb_sigma_delta_u.shape)
            print(seq_mb_sigma_delta_v)
            print(seq_mb_sigma_delta_v.shape)

            sum_seq_mb_mu_dist = np.sqrt((np.square(all_mb_u_np[i]) + np.square(all_mb_v_np[i])))
            for j in range(1, k):
                sum_seq_mb_mu_dist = sum_seq_mb_mu_dist + np.sqrt((np.square(all_mb_u_np[i+j]) + np.square(all_mb_v_np[i+j])))
            seq_mb_mu_dist = (1 / k) * sum_seq_mb_mu_dist

            print("####### Mean Magnitude #########")
            print(i)
            print(seq_mb_mu_dist)
            print(seq_mb_mu_dist.shape)

            print("####### MIN/MAX values #########")
            print(np.min(seq_mb_mu_dist))
            print(np.max(seq_mb_mu_dist))
            print(np.min(np.sqrt(np.square(seq_mb_sigma_delta_u) + np.square(seq_mb_sigma_delta_v))))
            print(np.max(np.sqrt(np.square(seq_mb_sigma_delta_u) + np.square(seq_mb_sigma_delta_v))))

            significance_condition1 = seq_mb_mu_dist > t1
            consistency_condition2 = np.sqrt(np.square(seq_mb_sigma_delta_u) + np.square(seq_mb_sigma_delta_v)) < t2
            final_condition = np.logical_and(significance_condition1, consistency_condition2)

            print("####### Conditions #########")
            print(significance_condition1)
            print(significance_condition1.shape)
            print(np.unique(significance_condition1, return_counts=True))
            print(consistency_condition2)
            print(consistency_condition2.shape)
            print(np.unique(consistency_condition2, return_counts=True))
            print(final_condition)
            print(final_condition.shape)
            print(np.unique(final_condition, return_counts=True))
            ''''''
            all_filter_masks_l.append(final_condition)

        all_filter_masks_np = np.array(all_filter_masks_l)
        print(all_filter_masks_np.shape)
        all_seq_mb_delta_u_np = np.array(all_seq_mb_delta_u_l)
        print(all_seq_mb_delta_u_np.shape)
        all_seq_mb_delta_v_np = np.array(all_seq_mb_delta_v_l)
        print(all_seq_mb_delta_v_np.shape)

        motion_l = []
        for i in range(0, len(all_mb_u_np) - n - k):
            filter_mask = all_filter_masks_np[i + 0]
            mvi_u = all_mb_u_np[i + 0][filter_mask == True]
            mvi_v = all_mb_v_np[i + 0][filter_mask == True]
            mvi_ang = all_mb_ang_np[i + n][filter_mask == True]
            mvi_u_sum = np.sum(np.abs(mvi_u))
            mvi_v_sum = np.sum(np.abs(mvi_v))

            mvi_u_cnt = len(mvi_u)
            mvi_v_cnt = len(mvi_v)

            mv_u = all_mb_u_np[i + 0][filter_mask == False]
            mv_v = all_mb_v_np[i + 0][filter_mask == False]
            mv_u_sum = np.sum(np.abs(mv_u))
            mv_v_sum = np.sum(np.abs(mv_v))
            mv_u_cnt = len(mv_u)
            mv_v_cnt = len(mv_v)

            print(mvi_u_sum)
            print(mvi_v_sum)
            print(mvi_u_cnt)
            print(mvi_v_cnt)
            print(mv_u_sum)
            print(mv_v_sum)
            print(mv_u_cnt)
            print(mv_v_cnt)

            if (mvi_u_sum < mv_u_sum * mvi_mv_ratio) and (mvi_v_sum < mv_v_sum * mvi_mv_ratio) and \
               (mvi_u_cnt < mv_u_cnt * mvi_mv_ratio) and (mvi_v_cnt < mv_v_cnt * mvi_mv_ratio):
                print("STATIC")
                motion_l.append("NA")
            else:
                print("OTHERS")
                # calculate angle for region with n frames
                print(f'DEBUG: {np.mean(mvi_ang)}')
                #print(f'DEBUG: {np.mean(mvi_mag)}')
                mean_ang = np.mean(mvi_ang)
                if ((mean_ang > 45 and mean_ang < 135) or (mean_ang > 225 and mean_ang < 315)):
                    class_name = "TILT"
                elif ((mean_ang > 135 and mean_ang < 225) or
                      (mean_ang > 315 and mean_ang < 360) or
                      (mean_ang > 0 and mean_ang < 45)):
                    class_name = "PAN"
                else:
                    class_name = "NA"
                motion_l.append(class_name)

            for j in range(1, n):
                filter_mask = all_filter_masks_np[i + j]

                mvi_u = all_mb_u_np[i + j][filter_mask == True]
                mvi_v = all_mb_v_np[i + j][filter_mask == True]
                mvi_ang = all_mb_ang_np[i + n][filter_mask == True]
                mvi_u_sum = mvi_u_sum + np.sum(np.abs(mvi_u))
                mvi_v_sum = mvi_v_sum + np.sum(np.abs(mvi_v))

                mv_u = all_mb_u_np[i + j][filter_mask == False]
                mv_v = all_mb_v_np[i + j][filter_mask == False]
                mv_u_sum = mv_u_sum + np.sum(np.abs(mv_u))
                mv_v_sum = mv_v_sum + np.sum(np.abs(mv_v))

                print(mvi_u_sum)
                print(mvi_v_sum)
                print(mv_u_sum)
                print(mv_v_sum)

                if (mvi_u_sum < mv_u_sum * mvi_mv_ratio) and (mvi_v_sum < mv_v_sum * mvi_mv_ratio) and \
                   (mvi_u_cnt < mv_u_cnt * mvi_mv_ratio) and (mvi_v_cnt < mv_v_cnt * mvi_mv_ratio):
                    print("STATIC")
                    motion_l.append("NA")
                else:
                    print("OTHERS")
                    print(f'DEBUG: {np.mean(mvi_ang)}')
                    #print(f'DEBUG: {np.mean(mvi_mag)}')
                    mean_ang = np.mean(mvi_ang)
                    if ((mean_ang > 45 and mean_ang < 135) or (mean_ang > 225 and mean_ang < 315)):
                        class_name = "TILT"
                    elif ((mean_ang > 135 and mean_ang < 225) or
                          (mean_ang > 315 and mean_ang < 360) or
                          (mean_ang > 0 and mean_ang < 45)):
                        class_name = "PAN"
                    else:
                        class_name = "NA"
                    motion_l.append(class_name)

        if (len(motion_l) <= 0):
            motion_l.append("NA")
        motion_np = np.array(motion_l)

        class_names, class_dist = np.unique(motion_np, return_counts=True)
        print(class_names)
        print(class_dist)

        idx = np.argmax(class_dist, axis=0)
        print(class_names[idx])
        final_class_prediction = class_names[idx]

        '''
        # visualize vectors and frames
        for i in range(0, len(all_filter_masks_np)):
            frame_rgb = cv2.cvtColor(frames_np[i], cv2.COLOR_GRAY2RGB)
            for s in range(0, len(all_mb_u_np[i])):
                # switch color
                if (all_filter_masks_np[i][s]):
                    point_color = (0, 0, 0)
                    line_color = (0, 255, 0)
                else:
                    point_color = (0, 0, 255)
                    line_color = (0, 0, 255)

                cv2.circle(frame_rgb, center=(all_mb_x_np[i][s], all_mb_y_np[i][s]), radius=1, thickness=1,
                           color=point_color)
                cv2.line(frame_rgb, pt1=(all_mb_x_np[i][s], all_mb_y_np[i][s]),
                         pt2=(int(all_mb_x_np[i][s] + all_mb_u_np[i][s]), int(all_mb_y_np[i][s] + all_mb_v_np[i][s])),
                         thickness=1,
                         color=line_color)

            win_name = "orig+vectors"  # 1. use var to specify window name everywhere
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # 2. use 'normal' flag
            cv2.imshow(win_name, frame_rgb)
            cv2.resizeWindow(win_name, frame_rgb.shape[0], frame_rgb.shape[1])
            s = cv2.waitKey(100)
        '''

        return final_class_prediction

    def predict_final_result_v3(self, mag_np, ang_np, u_np, v_np):
        # print(type(mag_l_n))
        # print(len(mag_l_n))
        # exit()

        print(ang_np.shape)
        print(mag_np.shape)
        print(u_np.shape)
        print(v_np.shape)

        mean_mag = np.mean(mag_np)
        mean_ang = np.mean(ang_np)

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude
        if ((mean_ang > 45 and mean_ang < 135) or (mean_ang > 225 and mean_ang < 315)) and (mean_mag > th):
            class_name = "TILT"
        elif ((mean_ang > 135 and mean_ang < 225) or
              (mean_ang > 315 and mean_ang < 360) or
              (mean_ang > 0 and mean_ang < 45)) and (mean_mag > th):
            class_name = "PAN"
        else:
            class_name = "NA"

        print("mean mag: " + str(mean_mag))
        print("mean angle: " + str(mean_ang))
        print("predicted class_name (angles): " + str(class_name))
        return class_name

    def predict_final_result_NEW(self, mag_np, ang_np, u_np, v_np):
        # print(type(mag_l_n))
        # print(len(mag_l_n))
        # exit()

        print(ang_np.shape)
        print(mag_np.shape)
        print(u_np.shape)
        print(v_np.shape)

        '''
        # plt.figure(1)
        fig, axs = plt.subplots(4)
        axs[0].plot(np.arange(len(mag_np)), mag_np)
        axs[1].plot(np.arange(len(ang_np)), ang_np)
        axs[1].set_ylim([-400, 400])
        axs[2].plot(np.arange(len(u_np)), u_np)
        axs[3].plot(np.arange(len(v_np)), v_np)
        plt.grid(True)
        plt.show()

        exit()
        '''

        mean_mag = np.mean(mag_np)
        mean_ang = np.mean(ang_np)
        x_comp_n = np.nan_to_num(np.mean(u_np))
        y_comp_n = np.nan_to_num(np.mean(v_np))

        print("x_sum: " + str(x_comp_n))
        print("y_sum: " + str(y_comp_n))

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude
        #th = 0.7
        mag_condition_pan = abs(x_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)
        mag_condition_tilt = abs(y_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)

        if ((mean_ang > 45 and mean_ang < 135) or (mean_ang > 225 and mean_ang < 315)) and (mean_mag > th):
            class_name = "TILT"
        elif ((mean_ang > 135 and mean_ang < 225) or
              (mean_ang > 315 and mean_ang < 360) or
              (mean_ang > 0 and mean_ang < 45)) and (mean_mag > th):
            class_name = "PAN"
        else:
            class_name = "NA"

        print("mag: " + str(mean_mag))
        print("angle: " + str(mean_ang))
        print("predicted class_name (angles): " + str(class_name))
        print("PAN: " + str(mag_condition_pan))
        print("TILT: " + str(mag_condition_tilt))

        '''
        if (class_name == "PAN" and mag_condition_pan == True):
            class_name = "PAN"
        elif (class_name == "TILT" and mag_condition_tilt == True):
            class_name = "TILT"
        else:
            class_name = "NA"
        '''
        print("overall predicted class_name: " + str(class_name))

        return class_name

    def window_filter(self, data_np, window_size=10, alpha=1.5):
        data_mean_l = []
        center = int(window_size/2)
        for i in range(0, len(data_np)):
            if (i-center < 0):
                filtered = np.nan_to_num(np.mean(data_np[i:i+center]))
            else:
                filtered = np.nan_to_num(np.mean(data_np[i-center:i+center]))
            data_mean_l.append(filtered)
        data_mean_np = np.array(data_mean_l)
        return data_mean_np

    def outlier_filter(self, data_np, alpha=1.0):
        # print(type(data_np))
        # print(data_np.shape)
        data_std = np.std(data_np)
        data_mean = np.mean(data_np)
        anomaly_cut_off = data_std * alpha
        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off

        # Generate outliers
        outliers_idx = []
        filtered_data = []
        for j in range(0, len(data_np)):
            if (j < len(data_np) - 1):
                nxt = data_np[j + 1]
                curr = data_np[j]
                prv = data_np[j - 1]
            else:
                nxt = data_np[j]
                curr = data_np[j]
                prv = data_np[j - 1]

            if curr > upper_limit or curr < lower_limit:
                data_mean = (prv + nxt) / 2.0
                filtered_data.append(data_mean)
                outliers_idx.append(j)
            else:
                filtered_data.append(curr)

        filtered_data_np = np.array(filtered_data)
        return filtered_data_np, outliers_idx

    def getBlock(self, frame=None, row=-1, col=-1, number_of_blocks=4):
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        block_w = int(frame_w / number_of_blocks)
        block_h = int(frame_h / number_of_blocks)

        start_idx_row = row * block_h
        start_idx_col = col * block_w
        stop_idx_row = row * block_h + block_h
        stop_idx_col = col * block_w + block_w

        block_center_x = start_idx_col + int(abs(start_idx_col - stop_idx_col) / 2)
        block_center_y = start_idx_row + int(abs(start_idx_row - stop_idx_row) / 2)

        if (len(frame.shape) == 3):
            frame_block = frame[start_idx_row:stop_idx_row, start_idx_col:stop_idx_col, :]
        elif (len(frame.shape) == 2):
            frame_block = frame[start_idx_row:stop_idx_row, start_idx_col:stop_idx_col]
        else:
            print("ERROR: something is wrong with the frame shape.")

        return frame_block, (block_center_x, block_center_y)

    def run(self):
        frames_np = self.video_frames

        print(frames_np.shape)

        filtered_mag_l_n = []
        filtered_angles_l_n = []
        number_of_features_l = []

        vector_x_sum_l = []
        vector_y_sum_l = []
        angles_l_n = []
        mag_l_n = []

        MIN_NUM_FEATURES = 200
        seed_idx = 0
        for i in range(1, len(frames_np)):
            # print("##########")

            prev_frame = frames_np[i-1]
            curr_frame = frames_np[i]

            #prev_frame = cv2.medianBlur(prev_frame, 5)
            #curr_frame = cv2.medianBlur(curr_frame, 5)

            #prev_frame_equ = cv2.equalizeHist(prev_frame)
            #curr_frame_equ = cv2.equalizeHist(curr_frame)

            #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            #prev_frame = cv2.filter2D(prev_frame, -1, kernel)
            #curr_frame = cv2.filter2D(curr_frame, -1, kernel)

            ''''''
            cv2.imshow("orig", curr_frame)
            k = cv2.waitKey(10)
            #continue

            distance_threshold = self.config_instance.distance_threshold
            kp_prev_list, kp_curr_list = self.feature_detector.getMatches(prev_frame, curr_frame, distance_threshold)

            #print("---")
            #print("number of features")
            #print(len(kp_curr_list))
            #print(len(kp_prev_list))

            if (len(kp_prev_list) == 0 or len(kp_curr_list) == 0):
                # mag_l_n.append([0, 0])
                # angles_l_n.append([0, 0])
                mag_l_n.append([0, 0])
                angles_l_n.append([0, 0])
                continue

            #curr_points = np.array(kp_curr_list).astype('float').reshape(-1, 1, 2)
            #prev_points = np.array(kp_prev_list).astype('float').reshape(-1, 1, 2)
            #mag_n, angle_n = self.compute_magnitude_angle(prev_points,
            #                                              curr_points)

            # np.array([[-1, -1.5], [1, -1]]), np.array([[-1, -1], [0.5, -1.5]])
            curr_points = np.array(kp_curr_list).astype('float')
            prev_points = np.array(kp_prev_list).astype('float')
            mag_n, angle_n = self.compute_mag_ang(prev_points, curr_points)


            number_of_features = len(curr_points)
            number_of_features_l.append(number_of_features)

            #filtered_mag_n, outlier_idx = self.outlier_filter(mag_n, alpha=1)
            #filtered_angle_n, outlier_idx = self.outlier_filter(angle_n, alpha=1)
            #filtered_mag_n1, outlier_idx = self.outlier_filter(mag_n, alpha=3)
            #filtered_angle_n = self.window_filter(data_np=filtered_angle_n, window_size=10)

            #print(len(curr_points))
            #print(len(prev_points))
            '''
            plt.plot(angle_n)
            plt.plot(filtered_angle_n)
            #plt.plot(filtered_mag_n1)
            #plt.ylim(-400, 400)
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            '''
            #print(np.std(filtered_angle_n))
            if (number_of_features <= MIN_NUM_FEATURES):
                seed_idx = i

            # print(mag_n)
            # print(angle_n)
            # angle_raw.append(angle_n.tolist())
            # mag_raw.append(mag_n.tolist())
            '''
            # mag_n = np.abs(mag_n)  # [:50])
            mag_n, outlier_idx = self.filter1D(mag_n, alpha=2.5)
            angles_cleanup = []
            angles_orig_np = angle_n
            for s in range(0, len(angles_orig_np)):
                if (outlier_idx == s):
                    angle_mean = (angles_orig_np[s - 1] + angles_orig_np[s + 1]) / 2.0
                    angles_cleanup.append(angle_mean)
                else:
                    angles_cleanup.append(angles_orig_np[s])
            angle_n = np.array(angles_cleanup)

            # print(mag_n)
            # print(angle_n)

            # mag_n = np.delete(mag_n, outliers_idx)

            vector_y = np.multiply(mag_n, np.sin(np.deg2rad(angle_n)))
            vector_x = np.multiply(mag_n, np.cos(np.deg2rad(angle_n)))

            vector_y_sum = vector_y.sum() / len(vector_y)
            vector_x_sum = vector_x.sum() / len(vector_x)
            # print("vector_y_sum: " + str(vector_y_sum))
            # print("vector_x_sum: " + str(vector_x_sum))

            vector_x_sum_l.append([0, vector_x_sum])
            vector_y_sum_l.append([0, vector_y_sum])
            # exit()
            '''
            mag_mean_n = np.median(mag_n)
            mag_l_n.append(mag_mean_n)

            #angle_n = np.abs(angle_n)  # [:50])
            angle_mean_n = np.median(angle_n)
            angles_l_n.append(angle_mean_n)

            #print(mag_n)
            #print(angle_n)


            print(mag_mean_n)
            print(angle_mean_n)
            '''
            plt.plot(np.arange(0, len(mag_l_n)), np.array(mag_l_n))
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            '''
            #filtered_angle_n = angles_l_n  #
            #filtered_angles_l_n.append(filtered_angle_n)

        # cv2.destroyAllWindows()
        #print(angles_l_n)

        #exit()

        '''
        data_np = np.array(angles_l_n)[:, 1:]
        window_size = 10
        data_mean_l = []
        data_std_l = []
        center = int(window_size / 2)

        data_normed_np = (data_np - np.min(data_np)) / (np.max(data_np) - np.min(data_np))
        for i in range(0, len(data_np)):
            if (i - center < 0):
                filtered = np.nan_to_num(np.mean(data_np[i:i + center]))
                std = np.nan_to_num(np.std(data_normed_np[i:i + center]))
            else:
                filtered = np.nan_to_num(np.mean(data_np[i - center:i + center]))
                std = np.nan_to_num(np.std(data_normed_np[i - center:i + center]))
            data_mean_l.append(filtered)
            data_std_l.append(std)
        data_mean_np = np.array(data_mean_l)
        data_std_np = np.array(data_std_l)

        filt_angle_np = data_mean_np
        std_angle_np = data_std_np
        '''

        #print(mag_l_n)
        #print(angles_l_n)
        exit()
        mag_np = np.array(mag_l_n)[:, 1:]

        th_std = 0.05
        th_perc = 15
        th_mag = 1.0
        #filt_angle_np[std_angle_np > th_std] = np.nan
        #mag_np[std_angle_np > th_std] = np.nan

        all = len(filt_angle_np)
        pred = len(filt_angle_np[std_angle_np <= th_std])
        percentage = np.round((pred/all) * 100, 3)
        #print(all)
        #print(pred)
        #print(percentage)

        if(percentage > th_perc):
            predicted_angle = np.nanmedian(filt_angle_np[std_angle_np <= th_std])
            predicted_mag = np.nanmedian(mag_np[std_angle_np <= th_std])
        else:
            predicted_angle = np.nan
            predicted_mag = np.nan

        print(percentage)
        print(predicted_mag)
        print(predicted_angle)

        if (self.config_instance.debug_flag == True):
            print(percentage)
            print(predicted_mag)
            print(predicted_angle)

            filt_angle_np[std_angle_np > th_std] = 400
            mag_np[std_angle_np > th_std] = -1
            fig, axs = plt.subplots(3)
            axs[0].plot(np.arange(len(mag_np)), mag_np)
            #axs[0].plot(np.arange(len(filt_angle_np)), filt_angle_np)
            axs[0].set_ylim([-2, 50])
            axs[1].plot(np.arange(len(filt_angle_np)), np.array(filt_angle_np))
            axs[1].set_ylim([-400, 600])
            axs[2].plot(np.arange(len(std_angle_np)), np.array(std_angle_np))
            axs[2].set_ylim([0, 1])
            plt.show()

            ''''''

        # conditions for different movement types
        tilt_up_condition = np.logical_and((predicted_angle > 45), (predicted_angle < 135))
        tilt_down_condition = np.logical_and((predicted_angle > 225), (predicted_angle < 315))
        pan_left_condition = np.logical_and((predicted_angle > 135), (predicted_angle < 225))
        pan_right_condition = np.logical_or(
            np.logical_and((predicted_angle >= 0), (predicted_angle < 45)),
            np.logical_and((predicted_angle > 315), (predicted_angle <= 360))
            )

        mag_condition = predicted_mag > th_mag

        print(tilt_up_condition)
        print(tilt_down_condition)
        print(pan_left_condition)
        print(pan_right_condition)
        print(mag_condition)
        if (self.config_instance.debug_flag == True):
            #if(self.config_instance)
            print(tilt_up_condition)
            print(tilt_down_condition)
            print(pan_left_condition)
            print(pan_right_condition)
            print(mag_condition)
            ''''''

        if (tilt_up_condition == True and mag_condition == True):
            #class_name = "TILT_UP"
            class_name = "TILT"
        elif (tilt_down_condition == True and mag_condition == True):
            #class_name = "TILT_DOWN"
            class_name = "TILT"
        elif (pan_left_condition == True and mag_condition == True):
            #class_name = "PAN_LEFT"
            class_name = "PAN"
        elif (pan_right_condition == True and mag_condition == True):
            #class_name = "PAN_RIGHT"
            class_name = "PAN"
        else:
            class_name = "NA"
        #print(class_name)
        #exit()
        return class_name
        #return mag_l_n, angles_l_n, vector_x_sum_l, vector_y_sum_l

    def predict_final_result(self, mag_l_n, angles_l_n, x_sum_l, y_sum_l, class_names):
        # print(type(mag_l_n))
        # print(len(mag_l_n))
        # exit()

        # calcualate final result
        angles_np = np.array(angles_l_n) 
        mag_np = np.array(mag_l_n)  

        x_comp = np.array(x_sum_l)  
        y_comp = np.array(y_sum_l)  
        print(angles_np.shape)
        print(mag_np.shape)
        print(x_comp.shape)
        print(y_comp.shape)

        x_comp_n = x_comp.sum() / len(x_comp)
        y_comp_n = y_comp.sum() / len(y_comp)

        print("x_sum: " + str(x_comp_n))
        print("y_sum: " + str(y_comp_n))

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude
        mag_condition_pan = abs(x_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2 )
        mag_condition_tilt = abs(y_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2 )
       

        '''
        # add filter - 1.5
        filtered_mag_n, outlier_idx = self.filter1D(mag_np[:, 1:], alpha=1.5)
        angles_cleanup = []
        angles_orig_np = angles_np[:, 1:]
        for s in range(0, len(angles_orig_np)):
            if(outlier_idx == s):
                angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                angles_cleanup.append(angle_mean)
            else:
                angles_cleanup.append(angles_orig_np[s])
        filtered_angles_np = np.array(angles_cleanup)  
        #filtered_angles_np = np.delete(angles_np[:, 1:], outlier_idx)

        # add filter - 1.5
        filtered_mag_n, outlier_idx = self.filter1D(filtered_mag_n, alpha=1.5)
        angles_cleanup = []
        angles_orig_np = angles_np[:, 1:]
        for s in range(0, len(angles_orig_np)):
            if(outlier_idx == s):
                angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                angles_cleanup.append(angle_mean)
            else:
                angles_cleanup.append(angles_orig_np[s])
        filtered_angles_np = np.array(angles_cleanup)  

        # add filter - 1.5
        filtered_mag_n, outlier_idx = self.filter1D(filtered_mag_n, alpha=1.5)
        angles_cleanup = []
        angles_orig_np = angles_np[:, 1:]
        for s in range(0, len(angles_orig_np)):
            if(outlier_idx == s):
                angle_mean = (angles_orig_np[s-1] + angles_orig_np[s+1]) / 2.0
                angles_cleanup.append(angle_mean)
            else:
                angles_cleanup.append(angles_orig_np[s])
        filtered_angles_np = np.array(angles_cleanup)  

        '''

        # add filter - 1.5
        filtered_angles_np, outlier_idx = self.filter1D(angles_np[:, 1:], alpha=1.5)
        
        # add filter - 1.5
        filtered_angles_np, outlier_idx = self.filter1D(filtered_angles_np, alpha=1.5)

        #filtered_angles_np = np.delete(angles_np[:, 1:], outlier_idx)

        
        #filtered_mag_n = self.filter1D(mag_np[:, 1:], alpha=1.5)
        #filtered_angles_np = self.filter1D(angles_np[:, 1:], alpha=1.5)

        #print(filtered_mag_n.shape)
        #print(np.mean(filtered_mag_n))
        #print(np.mean(mag_np[:, 1:]))
        #exit()
        #print(len(outlier_idx))
        #print(filtered_angles_np.shape)
        '''
        #filtered_angle_n, outlier_idx = self.filter1D(filtered_angles_np, alpha=2)
        #filtered_mag_n = np.delete(filtered_mag_n, outlier_idx)

        # calculate x - y components 
        vector_y = np.abs(np.multiply(filtered_mag_np, np.sin(np.deg2rad(filtered_angles_np))))
        vector_x = np.abs(np.multiply(filtered_mag_np, np.cos(np.deg2rad(filtered_angles_np))))

        '''

        b, bins, patches = plt.hist(filtered_angles_np, bins=8, range=[0, 360],
                                    cumulative=False)  # bins=None, range=None

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude

        percentage = 0.5  # ratio threshold between no-movement and movement
        class_names_n = ['PAN', 'TILT', 'TILT', 'PAN', 'PAN', 'TILT', 'TILT', 'PAN']

        DEBUG_VIS_FLAG = False
        if(DEBUG_VIS_FLAG == True):
            # plot angles over time (frames)
            fig, axs = plt.subplots(1)
            axs.plot(np.arange(len(mag_np[:, 1:])), mag_np[:, 1:])
            axs.plot(np.arange(len(filtered_mag_np)), filtered_mag_np)
            #b, bins, patches = axs.hist(filtered_angles_np, bins=8, range=[0,360], cumulative=False, alpha=0.7, rwidth=0.85)  #bins=None, range=None
            #minor_locator = AutoMinorLocator(2)
            #plt.gca().xaxis.set_minor_locator(minor_locator)
            #xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
            #xticks_labels = [ "[{:d}°-{:d}°]\n{:s}".format(int(value), int(bins[idx+1]), class_names_n[idx]) for idx, value in enumerate(bins[:-1])]
            #plt.xticks(xticks, labels=xticks_labels, rotation=45)
            #plt.grid(axis='y', alpha=0.75)
            plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.4)
            plt.savefig(self.config_instance.path_debug_results + "mag.pdf", dpi=500)
            plt.cla()
       
        #print("predicted mean magnitude: " + str(np.mean(filtered_mag_np)))
        #print("predicted vector x: " + str(np.mean(vector_x)))
        #print("predicted vector y: " + str(np.mean(vector_y)))
        #print("predicted median magnitude: " + str(np.median(filtered_mag_np)))
        #print("threshold magnitude: " + str(th))

        class_name = class_names_n[np.argmax(b)]
        print("predicted class_name (angles): " + str(class_name))
        print("PAN: " + str(mag_condition_pan))
        print("TILT: " + str(mag_condition_tilt))

        if (class_name == "PAN" and mag_condition_pan == True):
            class_name = "PAN"
        elif (class_name == "TILT" and mag_condition_tilt == True):
            class_name = "TILT"
        else:
            class_name = "NA"
        
        return class_name

    def filter1D(self, data_np, alpha=2.0):
        # print(type(data_np))
        # print(data_np.shape)
        data_std = np.std(data_np)
        data_mean = np.mean(data_np)
        anomaly_cut_off = data_std * alpha
        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off

        # Generate outliers
        outliers_idx = []
        filtered_data = []
        for j in range(0, len(data_np)):
            if (j < len(data_np) - 1):
                nxt = data_np[j + 1]
                curr = data_np[j]
                prv = data_np[j - 1]
            else:
                nxt = data_np[j]
                curr = data_np[j]
                prv = data_np[j - 1]

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
        #print(d.shape)
        mag = np.hypot(d[:, 0, 0], d[:, 0, 1])
        ang = np.round(np.arctan2(d[:, 0, 1], d[:, 0, 0]))

        # ang = np.round(np.arctan2(d[:, 0, 1], d[:, 0, 0])*180 / np.pi)

        return mag, ang

    def compute_mag_ang(self, vector_1, vector_2):
        y = vector_2[:, 1] - vector_1[:, 1]
        x = vector_2[:, 0] - vector_1[:, 0]

        mag = np.sqrt(x ** 2 + y ** 2)
        ang_2 = np.arctan2(y, (x + 0.0000000001))
        ang_degrees = np.degrees(ang_2)
        #np.arctan(y/(x + 0.0000000001))
        '''
        print(x)
        print(y)
        print(mag)
        print(ang_2)
        print(np.degrees(ang_2))
        print(ang_2)
        print(np.degrees(ang_2))
        '''

        ang_degrees[ang_degrees < 0] = 360 + ang_degrees[ang_degrees < 0]
        return mag, ang_degrees

        '''
        d = vector_2 - vector_1
        print(d)
        mag = np.hypot(d[:, 0], d[:, 1])
        ang = np.round(np.arctan2(d[:, 1], d[:, 0]))
        print(mag)
        print(ang)
        print(np.degrees(ang))
        '''
