from tracemalloc import start
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
import math
from scipy.stats import multivariate_normal


class OpticalFlow(object):
    def __init__(self, video_frames=None, algorithm=None, config_instance=None):
        self.video_frames = video_frames
        self.config_instance = config_instance

        self.number_of_blocks = 32

    def calculate_displacements_u_v(self):
        print("INFO: calculate dense optical flow for entire shot ...")
        frames_np = np.squeeze(self.video_frames)
        print(frames_np.shape)

        of_dense_instance = OpticalFlow_Dense()

        start_time1 = datetime.now()
        frm_u_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        frm_v_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        frm_mag_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        frm_ang_np = pymp.shared.array((frames_np.shape[0] - 1, frames_np.shape[1], frames_np.shape[2]), dtype='float32')
        with pymp.Parallel(4) as p:
            #p.print(p.num_threads, p.thread_num)
            for i in p.range(0, len(frames_np)-1):
                curr_frame = frames_np[i]
                nxt_frame = frames_np[i + 1]

                curr_frame = cv2.equalizeHist(curr_frame)
                nxt_frame = cv2.equalizeHist(nxt_frame)
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                curr_frame = cv2.filter2D(curr_frame, -1, kernel)
                nxt_frame = cv2.filter2D(nxt_frame, -1, kernel)
                frm_mag_np[i], frm_ang_np[i], frm_u_np[i], frm_v_np[i] = of_dense_instance.getFlow(curr_frame, nxt_frame)

        print(frm_mag_np.shape)
        print(frm_ang_np.shape)
        print(frm_u_np.shape)
        print(frm_v_np.shape)

        end_time1 = datetime.now()
        time_elapsed1 = end_time1 - start_time1
        print(time_elapsed1)

        return frm_mag_np, frm_ang_np, frm_u_np, frm_v_np

    def create_macro_blocks(self, frm_u_np, frm_v_np, frm_mag_np, frm_ang_np, VISUALIZE_ACTIVE_FLAG=False):
        # split into macro blocks (mb) - calculate one representative vector for each macro block (mean)
        start_time2 = datetime.now()
        
        all_mb_u_np = pymp.shared.array((frm_u_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')
        all_mb_v_np = pymp.shared.array((frm_v_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')
        all_mb_mag_np = pymp.shared.array((frm_mag_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')
        all_mb_ang_np = pymp.shared.array((frm_ang_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='float32')
        all_mb_pos_x_np = pymp.shared.array((frm_mag_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='int')
        all_mb_pos_y_np = pymp.shared.array((frm_ang_np.shape[0], self.number_of_blocks*self.number_of_blocks), dtype='int')

        with pymp.Parallel(4) as p:
            #p.print(p.num_threads, p.thread_num)
            for i in p.range(0, len(frm_u_np)):
            #for i in range(0, len(frm_u_np)):
                frm_u = frm_u_np[i]
                frm_v = frm_v_np[i]
                frm_mag = frm_mag_np[i]
                frm_ang = frm_ang_np[i]

                frm_mb_pos_x_l = []
                frm_mb_pos_y_l = []
                frm_mb_u_l = []
                frm_mb_v_l = []
                frm_mb_mag_l = []
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
                        frm_mb_pos_x_l.append(mb_center_x)
                        frm_mb_pos_y_l.append(mb_center_y)

                all_mb_u_np[i] = np.array(frm_mb_u_l)
                all_mb_v_np[i] = np.array(frm_mb_v_l)
                all_mb_mag_np[i] = np.array(frm_mb_mag_l)
                all_mb_ang_np[i] = np.array(frm_mb_ang_l)
                all_mb_pos_x_np[i] = np.array(frm_mb_pos_x_l)
                all_mb_pos_y_np[i] = np.array(frm_mb_pos_y_l)

        print(all_mb_mag_np.shape)
        print(all_mb_ang_np.shape)
        print(all_mb_u_np.shape)
        print(all_mb_v_np.shape)
        print(all_mb_pos_x_np.shape)
        print(all_mb_pos_y_np.shape)
        
        end_time2 = datetime.now()
        time_elapsed2 = end_time2 - start_time2
        print(time_elapsed2)

        if(VISUALIZE_ACTIVE_FLAG == True):
            frames_np = np.squeeze(self.video_frames)
            print(frames_np.shape)
            self.visualize_motion_vectors(frames_np, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np)

        return all_mb_mag_np, all_mb_ang_np, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np

    def visualize_motion_vectors(self, frames_np, u_np, v_np, x_np, y_np, mask=None):
        # visualize vectors and frames
        for i in range(0, len(frames_np) - 1):
            frame_rgb = cv2.cvtColor(frames_np[i], cv2.COLOR_GRAY2RGB)
            for s in range(0, len(u_np[i]) - 2):
                if(mask is None):
                    point_color = (0, 0, 255)
                    line_color = (0, 0, 255)
                else:
                    # switch color
                    if (mask[i][s]):
                        point_color = (0, 0, 0)
                        line_color = (0, 255, 0)
                    else:
                        point_color = (0, 0, 255)
                        line_color = (0, 0, 255)
                cv2.circle(frame_rgb, center=(x_np[i][s], y_np[i][s]), radius=1, thickness=1,
                           color=point_color)
                cv2.line(frame_rgb, pt1=(x_np[i][s], y_np[i][s]),
                         pt2=(int(x_np[i][s] + u_np[i][s]), int(y_np[i][s] + v_np[i][s])),
                         thickness=1,
                         color=line_color)

            win_name = "orig+vectors"  # 1. use var to specify window name everywhere
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # 2. use 'normal' flag
            cv2.imshow(win_name, frame_rgb)
            cv2.resizeWindow(win_name, frame_rgb.shape[0], frame_rgb.shape[1])
            s = cv2.waitKey(100)

    def filter_motion_vectors_of_interest_OLD(self, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np, k, n, t1, t2, VISUALIZE_ACTIVE_FLAG=True):
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

        if(VISUALIZE_ACTIVE_FLAG == True):
            frames_np = np.squeeze(self.video_frames)
            print(frames_np.shape)
            self.visualize_motion_vectors(frames_np, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np, all_filter_masks_np)
        
        return all_filter_masks_np

    def filter_motion_vectors_of_interest(self, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np, k, n, t1, t2, VISUALIZE_ACTIVE_FLAG=True):
        start_time3 = datetime.now()
        
        # calculate gradients of horizontal and vertical displacement (delta_u and delta_v) for each macro block (mb)
        all_mb_delta_u_l = []
        all_mb_delta_v_l = []
        for i in range(0, len(all_mb_u_np) - 1):
            u_curr = all_mb_u_np[i]
            v_curr = all_mb_v_np[i]
            u_next = all_mb_u_np[i + 1]
            v_next = all_mb_v_np[i + 1]

            delta_u = u_next - u_curr
            delta_v = v_next - v_curr
            all_mb_delta_u_l.append(delta_u)
            all_mb_delta_v_l.append(delta_v)
        all_mb_delta_u_np = np.array(all_mb_delta_u_l)
        all_mb_delta_v_np = np.array(all_mb_delta_v_l)

        print("####### DELTA U-V #########")
        print(all_mb_delta_u_np.shape)
        print(all_mb_delta_v_np.shape)


        # calculate mean magnitude for significance check
        
        all_mu_dist_l = []
        all_sigma_delta_u_l = []
        all_sigma_delta_v_l = []
        all_mu_delta_u_l = []
        all_mu_delta_v_l = []
        all_val_l = []
        for i in range(0, len(all_mb_delta_u_np) - k): 
            #print("significance check")
            a = np.square(all_mb_u_np[i:i+k, :])
            b = np.square(all_mb_v_np[i:i+k, :])
            c = np.sqrt(a + b)
            
            mu_dist_np = np.sum(c, axis=0) / k
            all_mu_dist_l.append(mu_dist_np)

            #print("consistence check")
            val_l = []
            sigma_delta_u_l = []
            sigma_delta_v_l = []
            for b in range(0, all_mb_delta_u_np.shape[1]):
                mu_mb_delta_u = np.mean(all_mb_delta_u_np[i:i+k, b])
                mu_mb_delta_v = np.mean(all_mb_delta_v_np[i:i+k, b])
                mu = np.array([mu_mb_delta_u, mu_mb_delta_v])
                
                t_mb_delta_u = np.expand_dims(all_mb_delta_u_np[i:i+k, b], axis=0)
                t_mb_delta_v = np.expand_dims(all_mb_delta_v_np[i:i+k, b], axis=0)
                X = np.concatenate((t_mb_delta_u,t_mb_delta_v), axis=0)
                cov = np.cov(X)
                #print(cov)
                #print(cov.shape)
           
                sigma_delta_u_np = cov[0][0]
                sigma_delta_v_np = cov[1][1]
                #print(sigma_delta_u_np)
                #print(sigma_delta_v_np)
                
                sigma_delta_u_l.append(sigma_delta_u_np)
                sigma_delta_v_l.append(sigma_delta_v_np)

                # sigma values
                val = np.sqrt(np.square(sigma_delta_u_np) + np.square(sigma_delta_v_np))
                # multivariate 
                #rv = multivariate_normal(mean=mu, cov=cov)
                #val = rv.pdf(x=all_mb_delta_u_np[i, b])
                val_l.append(val)

            val_np = np.array(val_l)
            all_val_l.append(val_np)
            
            all_sigma_delta_u_l.append(sigma_delta_u_l)
            all_sigma_delta_v_l.append(sigma_delta_v_l)
            #all_mu_delta_u_l.append(mu_delta_u_np)
            #all_mu_delta_v_l.append(mu_delta_v_np)


        all_mu_dist_np = np.array(all_mu_dist_l)
        print(all_mu_dist_np.shape)

        all_val_np = np.array(all_val_l)
        print(all_val_np.shape)
        print(all_val_np)
        print(np.min(all_val_np))
        print(np.max(all_val_np))
        print(np.mean(all_val_np))
        #exit()

        all_sigma_delta_u_np = np.array(all_sigma_delta_u_l)
        all_sigma_delta_v_np = np.array(all_sigma_delta_v_l)
        all_sigma_delta_u_np = np.nan_to_num(all_sigma_delta_u_np)
        all_sigma_delta_v_np = np.nan_to_num(all_sigma_delta_v_np)
        #all_mu_delta_u_np = np.array(all_mu_delta_u_l)
        #all_mu_delta_v_np = np.array(all_mu_delta_v_l)
        print(all_sigma_delta_u_np.shape)
        print(all_sigma_delta_v_np.shape)
        #print(all_mu_delta_u_np.shape)
        #print(all_mu_delta_v_np.shape)

        # condition1 significance check
        mask1 = all_mu_dist_np > t1
        print(np.min(all_mu_dist_np))
        print(np.max(all_mu_dist_np))
        print(np.mean(all_mu_dist_np))

        # condition2 consistence check
        mask2 = all_val_np < t2
        print(mask2)
        print(mask2.shape)

        # filter mask for mvi
        mvi_mask = np.logical_and(mask1, mask2)
        print(mvi_mask)

        if(VISUALIZE_ACTIVE_FLAG == True):
            frames_np = np.squeeze(self.video_frames)
            print(frames_np.shape)
            self.visualize_motion_vectors(frames_np, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np, mvi_mask)
        
        return mvi_mask

    def visualize_region_results(self, frames_np, all_motion_np, center_pos):
        # visualize region results vectors and frames
        for i in range(0, len(frames_np) - 1):
            frame_rgb = cv2.cvtColor(frames_np[i], cv2.COLOR_GRAY2RGB)
            for s in range(0, len(all_motion_np[i])):

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (center_pos[s][0] + int(170/2), center_pos[s][1] + int(170/2))
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                frame_rgb = cv2.putText(frame_rgb, all_motion_np[i][s], org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)

            win_name = "orig+mv"  # 1. use var to specify window name everywhere
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # 2. use 'normal' flag
            cv2.imshow(win_name, frame_rgb)
            cv2.resizeWindow(win_name, frame_rgb.shape[0], frame_rgb.shape[1])
            s = cv2.waitKey(100)

    def region_estimation(self, all_mb_u_np, all_mb_v_np, all_mb_mag_np, all_mb_ang_np, mvi_mask, n, k, mvi_mv_ratio, VISUALIZE_ACTIVE_FLAG=False):
        all_mb_u_np = all_mb_u_np[:-1, :]
        all_mb_v_np = all_mb_v_np[:-1, :]
        all_mb_ang_np = all_mb_ang_np[:-1, :]

        print(all_mb_u_np.shape)
        print(mvi_mask.shape)
        print(all_mb_ang_np.shape)
        print(all_mb_mag_np.shape)
        
        blocks = [[0,113], [113,226], [226,339], [339,452], [452,565], [565,678], [678,791], [791,904], [904,1024]]
        vis_blocks = [[0,0], [0,170], [0,340], [170,0], [170,170], [170,340], [340,0], [340,170], [340,340]]
        
        motion_l = []
        all_pc_list = []
        for j in range(0, len(all_mb_u_np) - n - k):
            print("###### result per frame ######")
            local_motion_l = []
            pc_list = []
            for block in blocks:
                #print(block)
                start = block[0]
                stop = block[1]
            
                mb_u_region_bl_np = all_mb_u_np[j:j+n, start:stop]
                mb_v_region_bl_np = all_mb_v_np[j:j+n, start:stop]
                mb_ang_region_bl_np = all_mb_ang_np[j:j+n, start:stop]
                mb_mag_region_bl_np = all_mb_mag_np[j:j+n, start:stop]
                mvi_mask_region_bl_np = mvi_mask[j:j+n, start:stop]

                #mag_mvi = np.mean(np.abs(mb_mag_region_bl_np[mvi_mask_region_bl_np == True]))
                #mag_mv = np.mean(np.abs(mb_mag_region_bl_np))
                #final_condition = mag_mvi < mag_mv * 0.1

                u_res1 = np.sum(np.abs(mb_u_region_bl_np[mvi_mask_region_bl_np == True]))
                u_res2 = np.sum(np.abs(mb_u_region_bl_np ))
                v_res1 = np.sum(np.abs(mb_v_region_bl_np[mvi_mask_region_bl_np == True]))
                v_res2 = np.sum(np.abs(mb_v_region_bl_np ))
                condition_part_a = u_res1 < u_res2 * mvi_mv_ratio
                condition_part_b = v_res1 < v_res2 * mvi_mv_ratio
                final_condition = condition_part_a and condition_part_b
                ''''''
                
                if(final_condition == True):
                    print("NA")
                    local_motion_l.append("NA")
                    pc_list.append(0)
                else:
                    #print("other movement")
                    #local_motion_l.append("other movement")

                    u_mvis = mb_u_region_bl_np[mvi_mask_region_bl_np == True]
                    v_mvis = mb_v_region_bl_np[mvi_mask_region_bl_np == True]
                    ang_np = mb_ang_region_bl_np[mvi_mask_region_bl_np == True]
                    #print(ang_np)

                    u_mvis = np.expand_dims(u_mvis, axis=1)
                    v_mvis = np.expand_dims(v_mvis, axis=1)

                    A = np.concatenate((u_mvis, v_mvis), axis=1)
                    #print(A .shape)
                    u, s, vh = np.linalg.svd(A)
                    #print(f'u: {u}\n')
                    #print(f's: {s}\n')
                    #print(f'vh: {vh}\n')

                    pc_of_region = vh[:,:1].squeeze()
                    #print(f'pc_of_region: {pc_of_region}')

                    # range [-pi, pi]
                    #theta_region = np.arctan2(pc_of_region[1], pc_of_region[0])
                    #theta_region = np.degrees(theta_region)
                    if(pc_of_region[1] >= 0):
                        theta_region = np.degrees(np.arccos(np.dot([1, 0], pc_of_region.T)))
                    else:
                        theta_region = 360 - np.degrees(np.arccos(np.dot([1, 0], pc_of_region.T)))

                    
                    #print(f'theta_region: {theta_region}')


                    #print(f'theta_region: {np.arctan2(pc_of_region[1], pc_of_region[0])}')
                    #theta_region_degrees = np.degrees(theta_region)
                    pc_list.append(theta_region)


                    if ((theta_region > 45 and theta_region < 135) or (theta_region > 225 and theta_region < 315)):
                        class_name = "TILT"
                    elif ((theta_region > 135 and theta_region < 225) or
                          (theta_region > 315 and theta_region < 360) or
                          (theta_region > 0 and theta_region < 45)):
                        class_name = "PAN"
                    else:
                        class_name = "NA"
                    print(class_name)
                    print(theta_region)
                    local_motion_l.append(class_name)

                    #plt.figure()
                    #plt.scatter(u_mvis, v_mvis)
                    #plt.show()
                    #exit()
                    #print(f'theta_region(degrees): {theta_region_degrees}')
        
            motion_l.append(local_motion_l)
            all_pc_list.append(pc_list)
        
        all_motion_np = np.array(motion_l)
        print(all_motion_np)
        print(all_motion_np.shape)
        print(np.unique(all_motion_np, return_counts=True))

        print(np.array(all_pc_list))
        print(np.array(all_pc_list).shape)

        region_block = np.array(all_pc_list)
        print(region_block)
        print(region_block.shape)

        # visualize region results vectors and frames
        if(VISUALIZE_ACTIVE_FLAG == True):
            frames_np = np.squeeze(self.video_frames)
            self.visualize_region_results(frames_np, all_motion_np, vis_blocks)
        ''''''

        return all_motion_np


    def runDense(self, vid_name, shot_id, shot_start, shot_end):
        frames_np = np.squeeze(self.video_frames)
        print(frames_np.shape)

        if(len(frames_np) > 10000):
            print("WARNING: shot is very very long! --> skip")
            class_name = "NA"
            return class_name, []

        #calculate dense optical flow vectors (u, v, mag, ang)
        frm_mag_np, frm_ang_np, frm_u_np, frm_v_np = self.calculate_displacements_u_v()

        # split into macro blocks (mb)
        all_mb_mag_np, all_mb_ang_np, all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np = self.create_macro_blocks(frm_u_np, frm_v_np, frm_mag_np, frm_ang_np)
        
        # motion vector of interest detection (MVI)
        k = self.config_instance.mvi_window_size
        n = self.config_instance.region_window_size
        t1 = self.config_instance.threshold_significance
        t2 = self.config_instance.threshold_consistency
        mvi_mv_ratio = self.config_instance.mvi_mv_ratio

        #self.filter_motion_vectors_of_interest(all_mb_u_np, all_mb_v_np, k, n, t1, t2)
        mvi_mask = self.filter_motion_vectors_of_interest_OLD(all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np, k, n, t1, t2, VISUALIZE_ACTIVE_FLAG=False)
        #mvi_mask = self.filter_motion_vectors_of_interest(all_mb_u_np, all_mb_v_np, all_mb_pos_x_np, all_mb_pos_y_np, k, n, t1, t2, VISUALIZE_ACTIVE_FLAG=False)

        # region estimation
        all_motion_np = self.region_estimation(all_mb_u_np, all_mb_v_np, all_mb_mag_np, all_mb_ang_np, mvi_mask, n, k, mvi_mv_ratio, VISUALIZE_ACTIVE_FLAG=False)

        # final region assessment
        print("final region assessment")
        all_final_motion_l = []
        for r in range(0, len(all_motion_np)):
            cls, cnts = np.unique(all_motion_np[r, :], return_counts=True)
            idx = np.argmax(cnts)
            final_class_name = cls[idx]
            all_final_motion_l.append(final_class_name)
           
        all_final_motion_np = np.array(all_final_motion_l)
        print(all_final_motion_np.shape)
        print(all_final_motion_np)

        # separate pan and tilt list
        print("#################################")
        print("split into pan and tilt list ... ")
        pan_list = self.find_movement_in_sequence(find_class="PAN", motion_np=all_final_motion_np)
        tilt_list = self.find_movement_in_sequence(find_class="TILT", motion_np=all_final_motion_np)
        pans_np = np.array(pan_list)
        tilts_np = np.array(tilt_list)
        print(pans_np)
        print(tilts_np)
                
        # condition A --> remove short movements  
        condition_a_flag = True
        if(condition_a_flag == True):   
            print("########################################")  
            print("Condition A: remove short movements ... ")
            min_length_of_motion = 20 #10
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
            max_length_of_gap = 5 #2

            if (len(tilts_np) >= 2):
                tilts_np = self.filter_movements_gaps(tilts_np, max_length_of_gap=max_length_of_gap)
                print(tilts_np)

            if (len(pans_np) >= 2):
                pans_np = self.filter_movements_gaps(pans_np, max_length_of_gap=max_length_of_gap)
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
        
        #exit()
        seq_dict_l = []
        if(final_movements_np is not None):    
            tmp1 = final_movements_np[:,:2].astype('int') + shot_start 
            tmp2 = final_movements_np[:,2:]
            final_movements_np = np.concatenate((tmp1, tmp2), axis=1)
            
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

        #exit()
        return class_name, seq_dict_l

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

        unique_cmcs = np.unique(motion_np)
        if(len(unique_cmcs) == 1 and unique_cmcs[0] == find_class):
            movement_list.append([start, stop, unique_cmcs[0]])
        else:
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
                elif(h == (len(motion_np) - 1) and state == "stop"):
                    #print("end of list")
                    stop = len(motion_np) - 1 
                    movement_list.append([start, stop, class_name_prev])
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
