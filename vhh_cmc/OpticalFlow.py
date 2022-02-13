import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from vhh_cmc.OpticalFlow_ORB import OpticalFlow_ORB
from vhh_cmc.OpticalFlow_SIFT import OpticalFlow_SIFT
from vhh_cmc.OpticalFlow_Dense import OpticalFlow_Dense
from datetime import datetime
import pymp


class OpticalFlow(object):
    def __init__(self, video_frames=None, algorithm=None, config_instance=None):
        self.video_frames = video_frames
        self.config_instance = config_instance

        self.number_of_blocks = 32

    def runDense(self, vid_name, shot_id, shot_start, shot_end):
        frames_np = np.squeeze(self.video_frames)
        print(frames_np.shape)

        if(len(frames_np) > 10000):
            print("WARNING: shot is very very long! --> skip")
            class_name = "NA"
            return class_name, []

        of_dense_instance = OpticalFlow_Dense()
        hsv = np.zeros_like(frames_np[0])
        hsv[..., 1] = 255

        # calcuate optical flow for all frames in the shot
        start_time1 = datetime.now()
        frm_u_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        frm_v_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        frm_mag_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        frm_ang_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        with pymp.Parallel(4) as p:
            #p.print(p.num_threads, p.thread_num)
            for i in p.range(0, len(frames_np)-1):

                # The parallel print function takes care of asynchronous output.
                #p.print('Yay! {} done!'.format(i))

                #frm_u_l = []
                #frm_v_l = []
                #frm_ang_l = []
                #for i in range(0, len(frames_np)-step_size, step_size):
                curr_frame = frames_np[i]
                nxt_frame = frames_np[i + 1]

                #curr_frame = cv2.equalizeHist(curr_frame)
                #nxt_frame = cv2.equalizeHist(nxt_frame)

                #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                #curr_frame = cv2.filter2D(curr_frame, -1, kernel)
                #nxt_frame = cv2.filter2D(nxt_frame, -1, kernel)

                frm_mag_np[i], frm_ang_np[i], frm_u_np[i], frm_v_np[i] = of_dense_instance.getFlow(curr_frame, nxt_frame)
                #frm_mag, frm_ang, frm_u, frm_v = of_dense_instance.getFlow(curr_frame, nxt_frame)
                #frm_u_l.append(frm_u)
                #frm_v_l.append(frm_v)
                #frm_ang_l.append(frm_ang)

        #frm_u_np = np.array(frm_u_l)
        #frm_v_np = np.array(frm_v_l)
        #frm_ang_np = np.array(frm_ang_l)
        print(frm_ang_np.shape)
        print(frm_u_np.shape)
        print(frm_v_np.shape)

        end_time1 = datetime.now()
        time_elapsed1 = end_time1 - start_time1
        print(time_elapsed1)

        # split into blocks
        start_time2 = datetime.now()
        all_mb_u_l = []
        all_mb_v_l = []
        all_mb_ang_l = []

        all_mb_u_np = pymp.shared.array((frm_u_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')
        all_mb_v_np = pymp.shared.array((frm_v_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')
        all_mb_ang_np = pymp.shared.array((frm_ang_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')

        with pymp.Parallel(4) as p:
            #p.print(p.num_threads, p.thread_num)
            for i in p.range(0, len(frm_u_np)):
            #for i in range(0, len(frm_u_np)):
                frm_u = frm_u_np[i]
                frm_v = frm_v_np[i]
                frm_ang = frm_ang_np[i]

                frm_mb_u_l = []
                frm_mb_v_l = []
                frm_mb_ang_l = []
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
                        mb_ang, (mb_center_x, mb_center_y) = self.getBlock(
                            frame=frm_ang,
                            row=r,
                            col=c,
                            number_of_blocks=self.number_of_blocks)

                        frm_mb_u_l.append(np.mean(mb_u))
                        frm_mb_v_l.append(np.mean(mb_v))
                        frm_mb_ang_l.append(np.mean(mb_ang))

                all_mb_u_np[i] = np.array(frm_mb_u_l)
                all_mb_v_np[i] = np.array(frm_mb_v_l)
                all_mb_ang_np[i] = np.array(frm_mb_ang_l)

            #all_mb_u_l.append(np.array(frm_mb_u_l))
            #all_mb_v_l.append(np.array(frm_mb_v_l))
            #all_mb_ang_l.append(np.array(frm_mb_ang_l))

        #all_mb_u_np = np.array(all_mb_u_l)
        #all_mb_v_np = np.array(all_mb_v_l)
        #all_mb_ang_np = np.array(all_mb_ang_l)
        print(all_mb_ang_np.shape)
        print(all_mb_u_np.shape)
        print(all_mb_v_np.shape)
        end_time2 = datetime.now()
        time_elapsed2 = end_time2 - start_time2
        print(time_elapsed2)
        #exit()

        # cleanup allocated arrays
        #del frm_u_l
        #del frm_v_l
        #del frm_ang_l
        frm_u_np = None
        frm_v_np = None
        frm_ang_np = None

        start_time3 = datetime.now()
        k = self.config_instance.mvi_window_size
        n = self.config_instance.region_window_size
        t1 = self.config_instance.threshold_significance
        t2 = self.config_instance.threshold_consistency
        mvi_mv_ratio = self.config_instance.mvi_mv_ratio

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

        end_time3 = datetime.now()
        time_elapsed3 = end_time3 - start_time3
        print('(DEBUG 1)Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed1))
        print('(DEBUG 2)Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed2))
        print('(DEBUG 3)Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed3))
        #exit()

        all_motion_l = []
        for i in range(0, len(all_mb_u_np) - n - k):
            motion_l = []


            # calculate ratio - initial frame at position 0 
            filter_mask = all_filter_masks_np[i]
            mvi_u = all_mb_u_np[i][filter_mask == True]
            mvi_v = all_mb_v_np[i][filter_mask == True]
            mvi_ang = all_mb_ang_np[i][filter_mask == True]
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

            '''
            print(mvi_u_sum)
            print(mvi_v_sum)
            print(mvi_u_cnt)
            print(mvi_v_cnt)
            print(mv_u_sum)
            print(mv_v_sum)
            print(mv_u_cnt)
            print(mv_v_cnt)
            '''

            if (mvi_u_sum < mv_u_sum * mvi_mv_ratio) and (mvi_v_sum < mv_v_sum * mvi_mv_ratio) and \
               (mvi_u_cnt < mv_u_cnt * mvi_mv_ratio) and (mvi_v_cnt < mv_v_cnt * mvi_mv_ratio):
                #print("STATIC")
                motion_l.append("NA")
            else:
                #print("OTHERS")
                # calculate angle for region with n frames
                #print(f'DEBUG 1: {np.mean(mvi_ang)}')
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
            
            # calculate ratio - frame position 1 to position n-1 
            for j in range(1, n):
                filter_mask = all_filter_masks_np[i + j]

                mvi_u = all_mb_u_np[i + j][filter_mask == True]
                mvi_v = all_mb_v_np[i + j][filter_mask == True]
                mvi_ang = all_mb_ang_np[i + j][filter_mask == True]
                mvi_u_sum = mvi_u_sum + np.sum(np.abs(mvi_u))
                mvi_v_sum = mvi_v_sum + np.sum(np.abs(mvi_v))

                mv_u = all_mb_u_np[i + j][filter_mask == False]
                mv_v = all_mb_v_np[i + j][filter_mask == False]
                mv_u_sum = mv_u_sum + np.sum(np.abs(mv_u))
                mv_v_sum = mv_v_sum + np.sum(np.abs(mv_v))

                #print(mvi_u_sum)
                #print(mvi_v_sum)
                #print(mv_u_sum)
                #print(mv_v_sum)

                if (mvi_u_sum < mv_u_sum * mvi_mv_ratio) and (mvi_v_sum < mv_v_sum * mvi_mv_ratio) and \
                   (mvi_u_cnt < mv_u_cnt * mvi_mv_ratio) and (mvi_v_cnt < mv_v_cnt * mvi_mv_ratio):
                    #print("STATIC")
                    motion_l.append("NA")
                else:
                    #print("OTHERS")
                    #print(f'DEBUG: {np.mean(mvi_ang)}')
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

            region_motion_names, region_motion_dist = np.unique(motion_np, return_counts=True)
            idx = np.argmax(region_motion_dist, axis=0)
            region_class_prediction = region_motion_names[idx]
            all_motion_l.append(region_class_prediction)

        all_motion_np = np.array(all_motion_l)
        print(all_motion_l)
        print(all_motion_np.shape)

        if(False):
            # separate pan and tilt list
            print("#################################")
            print("split into pan and tilt list ... ")
            pan_list = self.find_movement_in_sequence(find_class="PAN", motion_np=all_motion_np)
            tilt_list = self.find_movement_in_sequence(find_class="TILT", motion_np=all_motion_np)
            pans_np = np.array(pan_list)
            tilts_np = np.array(tilt_list)
            print(pans_np)
            print(tilts_np)
            
            # condition A --> remove short movements  
            condition_a_flag = True
            if(condition_a_flag == True):   
                print("########################################")  
                print("Condition A: remove short movements ... ")
                min_length_of_motion = 15
                pans_np = self.filter_short_movements(pans_np, min_length_of_motion=min_length_of_motion)
                tilts_np = self.filter_short_movements(tilts_np, min_length_of_motion=min_length_of_motion)
                print(pans_np)
                print(tilts_np)
            else:
                print("#######################################################")  
                print("Condition A: remove short movements --> NOT ACTIVE ... ")
                print()

            # Condition B --> Find Gaps
            condition_b_flag = True
            if(condition_b_flag == True):   
                print("######################################")
                print("Condition B: find and filter gaps ... ")
                max_length_of_gap = 5

                if (len(tilts_np) >= 2):
                    tilts_np = self.filter_movements_gaps(tilts_np, max_length_of_gap=max_length_of_gap)
                    pans_np = self.filter_movements_gaps(pans_np, max_length_of_gap=max_length_of_gap)
                    print(tilts_np)
                    print(pans_np)
            else:
                print("#######################################################")  
                print("Condition B: find and filter gaps --> NOT ACTIVE ... ")
                print()

        
            # map movment lists to shot boundaries
            final_movements_np = None
            if(len(tilts_np) > 0 and len(pans_np) > 0):
                final_movements_np = np.concatenate((tilts_np, pans_np), axis=0)
            elif(len(tilts_np) > 0 and len(pans_np) <= 0):
                final_movements_np = tilts_np
            elif(len(pans_np) > 0 and len(tilts_np) <= 0):
                final_movements_np = pans_np
            
            # TODO: save final movements numpy into file
            seq_dict_l = []
            if(final_movements_np is not None):    
                final_movements_np[:,:2] = final_movements_np[:,:2].astype('int') + shot_start 
                print(final_movements_np)
                #print(shot_start)
                #print(shot_end)

                # save sequences to disc
                import json
                
                for g in range(0, len(final_movements_np)):
                    seq_start = final_movements_np[g][0]
                    seq_stop = final_movements_np[g][1]
                    class_name = final_movements_np[g][2]
                    #line = [vid_name.split('.')[0], shot_id, seq_start, seq_stop, class_name]
                    seq_dict = {
                        "shotId": shot_id,
                        "start": seq_start,
                        "stop": seq_stop,
                        "cmcType": class_name
                    }
                    seq_dict_l.append(seq_dict)
            
                final_movements_duration_np = np.abs(final_movements_np[:,0:1].astype('int') - final_movements_np[:,1:2].astype('int'))
                final_movements_duration_np = np.concatenate((final_movements_duration_np, final_movements_np[:,2:]), axis=1)
                print(final_movements_duration_np)

                idx = np.where(final_movements_duration_np[:, 1:] == "PAN")[0]
                pan_duration = np.sum(final_movements_duration_np[idx, :1].astype('int'))
                print(pan_duration)

                idx = np.where(final_movements_duration_np[:, 1:] == "TILT")[0]
                tilt_duration = np.sum(final_movements_duration_np[idx, :1].astype('int'))
                print(tilt_duration)

                if(pan_duration >= tilt_duration): 
                    class_name = "PAN"
                elif(tilt_duration > pan_duration): 
                    class_name = "TILT"
                else:
                    class_name = "NA"
                print(class_name)
            else:
                class_name = "NA"
                print(class_name)

            return class_name, seq_dict_l
                        
        '''        
        plt.figure()
        plt.scatter(np.arange(len(all_motion_np)), all_motion_np)
        plt.show()
        '''
   
        ''''''
        class_names, class_dist = np.unique(all_motion_np, return_counts=True)
        print(class_names)
        print(class_dist)
   
        # final ratio check or class distribution
        all_detections = np.sum(class_dist)
        print(all_detections)

        class_dist_percentage = class_dist / all_detections
        print(class_dist_percentage)

        threshold = self.config_instance.active_threshold
        conditions = class_dist_percentage > threshold
        print(conditions)

        idx = np.where(conditions == True)[0]
        print(idx)
        class_names = class_names[idx]
        class_dist = class_dist[idx]
        print(class_names)
        print(class_dist)

        na_idx = np.where(class_names == "NA")[0]
        pan_idx = np.where(class_names == "PAN")[0]
        tilt_idx = np.where(class_names == "TILT")[0]
        print(na_idx)
        print(pan_idx)
        print(tilt_idx)
        print(len(na_idx))
        print(len(pan_idx))
        print(len(tilt_idx))

        if(len(na_idx) == 1 and len(pan_idx) == 0 and len(tilt_idx) == 0):
            print("only na")
            final_class_prediction = "NA"
        elif(len(na_idx) == 1 and len(pan_idx) == 1 and len(tilt_idx) == 0):
            print("only na and pans")
            idx = np.argmax(class_dist, axis=0)
            print(class_names[idx])
            final_class_prediction = class_names[idx]
            #final_class_prediction = "PAN"
        elif (len(na_idx) == 1 and len(pan_idx) == 0 and len(tilt_idx) == 1):
            print("only na and tilts")
            idx = np.argmax(class_dist, axis=0)
            print(class_names[idx])
            final_class_prediction = class_names[idx]
            #final_class_prediction = "TILT"
        elif(len(na_idx) == 0 and len(pan_idx) == 1 and len(tilt_idx) == 1):
            print("only pans and tilts")
            idx = np.argmax(class_dist, axis=0)
            print(class_names[idx])
            final_class_prediction = class_names[idx]
        elif(len(na_idx) == 0 and len(pan_idx) == 1 and len(tilt_idx) == 0):
            print("only pans")
            idx = np.argmax(class_dist, axis=0)
            print(class_names[idx])
            final_class_prediction = class_names[idx]
        elif(len(na_idx) == 0 and len(pan_idx) == 0 and len(tilt_idx) == 1):
            print("only tilts")
            idx = np.argmax(class_dist, axis=0)
            print(class_names[idx])
            final_class_prediction = class_names[idx]
        elif(len(na_idx) == 1 and len(pan_idx) == 1 and len(tilt_idx) == 1):
            print("na, pans and tilts")
            idx = np.argmax(class_dist, axis=0)
            print(class_names[idx])
            final_class_prediction = class_names[idx]
        else:
            print("na")
            final_class_prediction = "NA"

        print(final_class_prediction)
        

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

        return final_class_prediction, []

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

    def find_movement_in_sequence(self, find_class="TILT", motion_np=None):
        state = "start"
        start = 0
        stop = len(motion_np)
        class_name_prev = "NA"
        movement_list = []
        for h in range(0, len(motion_np)):
            #print(h)
            class_name = motion_np[h]

            # find start of movement
            if(class_name == find_class and state == "start"):
                #print("state: start")
                start = h
                state = "stop"
                
            elif(class_name_prev != class_name and state == "stop"):
                #print("state: stop")
                stop = h-1
                movement_list.append([start, stop, class_name_prev])
                state = "start"

            class_name_prev = class_name
        return movement_list

    def filter_short_movements(self, motion_np, min_length_of_motion=5):
        filtered_motion_list = []
        for h in range(0, len(motion_np)):
            start = int(motion_np[h][0])
            stop = int(motion_np[h][1])
            class_name = motion_np[h][2]
            if(stop < start):
                print("ERROR: something is wrong(DEBUG Point C)!!")
                exit()

            difference = stop - start
            if(difference >= min_length_of_motion):
                filtered_motion_list.append([start, stop, class_name])

        return np.array(filtered_motion_list)

    def filter_movements_gaps(self, motion_np, max_length_of_gap=5):
        m_l = []

        if (len(motion_np) <= 0):
            return np.array(m_l)

        start_seed = motion_np[0][0]
        stop_seed = motion_np[0][1]
        class_seed = motion_np[0][2]

        for h in range(1, len(motion_np)):
            start_nxt = motion_np[h][0]
            stop_nxt = motion_np[h][1]
            class_nxt = motion_np[h][2]

            difference = abs(int(stop_seed) - int(start_nxt))
            if(difference <= max_length_of_gap):
                stop_seed = stop_nxt
            else:
                m_l.append([start_seed, stop_seed, class_seed])
                start_seed = start_nxt
                stop_seed = stop_nxt
                class_seed = class_nxt

            if (h >= len(motion_np) - 1):
                m_l.append([start_seed, stop_seed, class_seed])
        return np.array(m_l)

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
