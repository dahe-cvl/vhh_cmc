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
        step_size = 2

        fig = plt.figure()
        gs = gridspec.GridSpec(nrows=1, ncols=2)

        orig_mag_l = []
        orig_angles_l = []

        for i in range(step_size, len(frames_np), step_size):
            prev_frame = frames_np[i - step_size]
            curr_frame = frames_np[i]

            prev_frame_equ = cv2.equalizeHist(prev_frame)
            curr_frame_equ = cv2.equalizeHist(curr_frame)

            mag, ang, u, v = of_dense_instance.getFlow(prev_frame, curr_frame)
            orig_mag_l.append(mag)
            orig_angles_l.append(ang)

            '''
            print(np.min(mag))
            print(np.max(mag))
            
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(curr_frame_equ, cmap='gray')

            #ax1 = fig.add_subplot(gs[0, 1])
            #ax1.hist(mag.flatten())

            ax1 = fig.add_subplot(gs[0, 1], projection='polar')
            plt.scatter(ang.flatten(), mag.flatten(), s=1)
            #plt.ylim(0, 15)
            #plt.xlim(0, 8)
            plt.draw()
            plt.pause(0.1)
            plt.cla()
            '''

            continue


            # print("################")
            block_u_mean_l = []
            block_v_mean_l = []

            new_block_mag_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))
            new_block_ang_mean_np = np.zeros((self.number_of_blocks, self.number_of_blocks))


            block_mag_mean_l = []
            block_ang_mean_l = []
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
                    block_mag, block_ang = cv2.cartToPolar(u_block, v_block)

                    # mag_mean = int(np.mean(mag_block))
                    # print(mag_mean)

                    block_mag_mean = np.mean(block_mag)
                    block_ang_mean = np.mean(np.degrees(block_ang))

                    new_block_mag_mean_np[r, c] = block_mag_mean
                    new_block_ang_mean_np[r, c] = block_ang_mean

                    block_mag_mean_l.append(block_mag_mean)
                    block_ang_mean_l.append(block_ang_mean)

                    u_mean = np.mean(u_block)
                    v_mean = np.mean(v_block)
                    block_u_mean_l.append(u_mean)
                    block_v_mean_l.append(v_mean)

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
                    # cv2.imshow("block " + str(r) + "-" + str(c), mag_block)

            block_mag_mean_np = np.array(block_mag_mean_l)
            block_ang_mean_np = np.array(block_ang_mean_l)
            print(block_mag_mean_np.shape)
            '''
            window_filter_block_mag_mean_np = self.window_filter(block_mag_mean_np,window_size=20)

            filtered_block_mag_mean_np1, outlier_idx = self.outlier_filter(block_mag_mean_np, alpha=1)
            filtered_block_mag_mean_np2, outlier_idx = self.outlier_filter(block_mag_mean_np, alpha=2)
            filtered_block_mag_mean_np3, outlier_idx = self.outlier_filter(block_mag_mean_np, alpha=3)
            print(filtered_block_mag_mean_np1.shape)

            plt.plot(np.arange(len(block_mag_mean_np)), block_mag_mean_np)
            plt.plot(np.arange(len(window_filter_block_mag_mean_np)), window_filter_block_mag_mean_np)
            #plt.plot(np.arange(len(filtered_block_mag_mean_np1)), filtered_block_mag_mean_np1)
            #plt.plot(np.arange(len(filtered_block_mag_mean_np2)), filtered_block_mag_mean_np2)
            #plt.plot(np.arange(len(filtered_block_mag_mean_np3)), filtered_block_mag_mean_np3)
            plt.show()
            '''
            #exit()

            #print(block_ang_mean_np.shape)
            new_blocks_mag_l.append(new_block_mag_mean_np)
            new_blocks_ang_l.append(new_block_ang_mean_np)
            #print(new_block_ang_mean_np.shape)
            #exit()

            #plt.plot(np.arange(len(block_ang_mean_np)), block_ang_mean_np)
            #plt.show()

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

        orig_mag_np = np.array(orig_mag_l)
        orig_angles_np = np.array(orig_angles_l)
        
        #print(orig_mag_np.shape)
        #print(orig_angles_np.shape)
        #print(orig_angles_np.flatten().shape)
        #plt.hist(orig_angles_np.flatten())
        b, bins, patches = plt.hist(orig_angles_np.flatten(), bins=8, range=[0, 360],
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

        print(b_perc_np)

        th_perc = 50.0

        if ( ((b_perc_np[0] + b_perc_np[7]) > th_perc) or ((b_perc_np[3] + b_perc_np[4]) > th_perc) ):
            print("PAN")
            print((b_perc_np[0] + b_perc_np[7]))
            print((b_perc_np[3] + b_perc_np[4]))
            class_name = "PAN"
        elif( ((b_perc_np[1] + b_perc_np[2]) > th_perc) or ((b_perc_np[5] + b_perc_np[6]) > th_perc)):
            print("TILT")
            class_name = "TILT"
        else:
            print("NA")
            class_name = "NA"

        return class_name

        exit()
        ''''''

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

        #x_filtered_mag_np = filtered_mag_np[filtered_mag_np > THRESHOLD1]
        x_filtered_mag_np = self.window_filter(mag_np[mag_np > th], window_size=10)
        x_filtered_ang_np = self.window_filter(ang_np[mag_np > th], window_size=10)
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

        return x_filtered_ang_np, filtered_u_np, filtered_v_np

    def predict_final_result_NEW(self, ang_np, u_np, v_np):
        # print(type(mag_l_n))
        # print(len(mag_l_n))
        # exit()

        mean_ang = np.mean(ang_np)
        x_comp_n = np.nan_to_num(np.mean(u_np))
        y_comp_n = np.nan_to_num(np.mean(v_np))

        print("x_sum: " + str(x_comp_n))
        print("y_sum: " + str(y_comp_n))

        th = self.config_instance.min_magnitude_threshold  # 2.0  # manual set threshold for magnitude
        #th = 0.7
        mag_condition_pan = abs(x_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)
        mag_condition_tilt = abs(y_comp_n) > th and abs(x_comp_n - y_comp_n) > ((abs(x_comp_n) + abs(y_comp_n)) / 2)

        if (mean_ang > 45 and mean_ang < 135) or (mean_ang > 225 and mean_ang < 315):
            class_name = "TILT"
        elif (mean_ang > 135 and mean_ang < 225) \
                or (mean_ang > 315 and mean_ang < 0) \
                or (mean_ang > 0 and mean_ang < 45):
            class_name = "PAN"
        else:
            class_name = "NA"

        print("angle: " + str(mean_ang))
        print("predicted class_name (angles): " + str(class_name))
        print("PAN: " + str(mag_condition_pan))
        print("TILT: " + str(mag_condition_tilt))

        if (class_name == "PAN" and mag_condition_pan == True):
            class_name = "PAN"
        elif (class_name == "TILT" and mag_condition_tilt == True):
            class_name = "TILT"
        else:
            class_name = "NA"

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

        filtered_mag_l_n = []
        filtered_angles_l_n = []
        number_of_features_l = []

        vector_x_sum_l = []
        vector_y_sum_l = []
        angles_l_n = []
        mag_l_n = []

        MIN_NUM_FEATURES = 500
        seed_idx = 0
        for i in range(1, len(frames_np)):
            # print("##########")

            prev_frame = frames_np[seed_idx]
            curr_frame = frames_np[i]

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

            curr_points = np.array(kp_curr_list).astype('float').reshape(-1, 1, 2)
            prev_points = np.array(kp_prev_list).astype('float').reshape(-1, 1, 2)
            mag_n, angle_n = self.compute_magnitude_angle(prev_points,
                                                          curr_points)

            number_of_features = len(curr_points)
            number_of_features_l.append(number_of_features)



            if (number_of_features <= MIN_NUM_FEATURES):
                seed_idx = i

            # print(mag_n)
            # print(angle_n)
            # angle_raw.append(angle_n.tolist())
            # mag_raw.append(mag_n.tolist())
            ''''''
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

            mag_mean_n = np.mean(mag_n)
            mag_l_n.append([0, mag_mean_n])

            angle_n = np.abs(angle_n)  # [:50])
            angle_mean_n = np.mean(angle_n)
            angles_l_n.append([0, angle_mean_n])

            filtered_angle_n = angles_l_n
            filtered_angles_l_n.append(filtered_angle_n)

        # cv2.destroyAllWindows()
        return mag_l_n, angles_l_n, vector_x_sum_l, vector_y_sum_l

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
        ang = np.round(np.degrees(np.arctan2(d[:, 0, 1], d[:, 0, 0])))

        # ang = np.round(np.arctan2(d[:, 0, 1], d[:, 0, 0])*180 / np.pi)

        return mag, ang
