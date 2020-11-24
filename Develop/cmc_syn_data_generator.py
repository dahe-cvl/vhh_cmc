import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


class CmcSynDataGenerator(object):
    def __init__(self, image_path, results_path):
        print("create instance of syn data generator for cmc")
        #self.image_path = "/caa/Homes01/dhelm/working/vhh/develop/example_images/"
        #self.results_path = "/caa/Homes01/dhelm/working/vhh/develop/example_output/"
        self.image_path = image_path
        self.results_path = results_path

    def createGlobalImageNoise(self, img=None, global_img_noise_type=None):
        # pixel-based video-flicker noise
        if(global_img_noise_type == "STANDARD"):
            noise_img = np.random.randint(-20, 20, size=(img.shape[0], img.shape[1], img.shape[2]))
        elif(global_img_noise_type == "STRONG"):
            noise_img = np.random.randint(-50, 50, size=(img.shape[0], img.shape[1], img.shape[2]))
        elif(global_img_noise_type == "OFF"):
            noise_img = np.zeros((img.shape[0], img.shape[1], img.shape[2])).astype('uint8')
        else:
            print("ERROR: select valid GLOBAL_IMAGE_NOISE_TYPE!")
            exit()

        img_noisy = img.copy()
        img_noisy = img_noisy + noise_img
        img_noisy[img_noisy > 255] = 255
        img_noisy[img_noisy < 0] = 0
        img_noisy = img_noisy.astype('uint8')

        img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)
        img_noisy_gray = cv2.cvtColor(img_noisy_gray, cv2.COLOR_GRAY2RGB)
        #cv2.imshow("noisy pattern", noise_img.astype('uint8'))
        #cv2.imshow("noisy", img_noisy)
        #k = cv2.waitKey(10)

        return img_noisy_gray


    def createMovement(self, img=None, bb_w=-1, bb_h=-1, bb_pos_x=-1, bb_pos_y=-1, 
                             movement_type=None, 
                             shaky_mode_type=None, 
                             global_img_noise_type=None,
                             magnitude_percentage=-1,
                             sequence_len=-1):
        frame_w = img.shape[1]
        frame_h = img.shape[0]

        if(bb_w >= frame_w or bb_h >= frame_h):
            print("ERROR: bounding box is too large!")
            print("frame size: (" + str(frame_h) + "|" + str(frame_w) + ")")
            print("bb size: (" + str(bb_h) + "|" + str(bb_w) + ")")
            exit()

        if(bb_pos_x >= frame_w or bb_pos_x < 0):
            print("ERROR: start x position of bounding box is not valid!")
            print("bb pos_x is not in range: " + "0  <= " + str(bb_pos_x) + " < " + str(frame_w))
            exit()

        if(bb_pos_y >= frame_h or bb_pos_y < 0):
            print("ERROR: start x position of bounding box is not valid!")
            print("bb pos_y is not in range: " + "0  <= " + str(bb_pos_y) + " < " + str(frame_h))
            exit()

        if(magnitude_percentage > 100 or magnitude_percentage < 0):
            print("ERROR: magnitude_percentage is not valid!")
            print("magnitude_percentage is not in range: " + "0  <= " + str(magnitude_percentage) + " <= 100%")
            exit()

        if(sequence_len < 1):
            print("ERROR: sequence_len is not valid!")
            print("sequence_len is not: " + str(sequence_len) + " > 1 frame")
            exit()


        # create x,y direction noise
        if(shaky_mode_type == "STANDARD"):
            noise_x = np.random.randint(-1, 1, size=img.shape[1])
            noise_y = np.random.randint(-1, 1, size=img.shape[0])
        elif(shaky_mode_type == "STRONG"):
            noise_x = np.random.randint(-3, 3, size=img.shape[1])
            noise_y = np.random.randint(-3, 3, size=img.shape[0])
        elif(shaky_mode_type == "OFF"):
            noise_x = np.zeros(img.shape[1]).astype('uint8')
            noise_y = np.zeros(img.shape[0]).astype('uint8')
        else:
            print("ERROR: select valid SHAKY_MODE_TYPE!")
            exit()

        # calculate step size based on magnitude percentage parameter    
        MAX_MAG_PX = 10     
        mag_px = int( (magnitude_percentage * MAX_MAG_PX) / 100.0 )
        step_size = mag_px
        if(step_size <= 0): step_size = 1
        if(step_size > MAX_MAG_PX): step_size = MAX_MAG_PX

        '''
        d = 1
        k = 2
        x = np.arange(5000)
        f_x = np.array(k*x + d)

        f_x_normalized = (f_x - np.min(f_x)) / (np.max(f_x) - np.min(f_x))
        f_x_normalized = np.round(np.interp(f_x, (f_x.min(), f_x.max()), (0, 100)))
        print(np.unique(f_x_normalized))
        print(f_x_normalized)
        '''

        if(movement_type == 'PAN_RIGHT'):
            # bounding box
            #bb_pos_x = 10
            #bb_pos_y = 10 
            #bb_w = 300
            #bb_h = 240            

            # sliding window in x direction  --> PAN RIGHT
            cropped_window_l = []
            for i in range(0, img.shape[1], step_size):
                if(bb_pos_x + noise_x[i] + i + bb_w >= img.shape[1]):
                    break

                img_with_rect = img.copy()
                pt1 = (bb_pos_x + noise_x[i] + i, bb_pos_y + noise_y[i])
                pt2 = (bb_pos_x + noise_x[i] + i + bb_w, bb_pos_y + noise_y[i] + bb_h)
                cropped_img = img_with_rect[bb_pos_y + noise_y[i]:bb_pos_y + noise_y[i] + bb_h, (bb_pos_x + noise_x[i] + i):(bb_pos_x + noise_x[i] + i + bb_w), :]
                cropped_noisy_img = self.createGlobalImageNoise(img=cropped_img, global_img_noise_type=global_img_noise_type)
                cropped_window_l.append(cropped_noisy_img)
                if(len(cropped_window_l) >= sequence_len):
                    break

            cropped_window_np = np.array(cropped_window_l)

        elif(movement_type == 'PAN_LEFT'):
            # bounding box
            #bb_pos_x = 500
            #bb_pos_y = 10 
            #bb_w = 300
            #bb_h = 240

            # sliding window in x direction  --> PAN LEFT
            cropped_window_l = []
            for i in range(0, img.shape[1], step_size):
                if(bb_pos_x + noise_x[i] - i <= 0):
                    break

                img_with_rect = img.copy()
                pt1 = (bb_pos_x + noise_x[i] - i, bb_pos_y + noise_y[i])
                pt2 = (bb_pos_x + noise_x[i] - i + bb_w, bb_pos_y + noise_y[i] + bb_h)
                cropped_img = img[(bb_pos_y + noise_y[i]):(bb_pos_y + noise_y[i] + bb_h), (bb_pos_x + noise_x[i] - i):(bb_pos_x + noise_x[i] - i + bb_w), :]
                cropped_noisy_img = self.createGlobalImageNoise(img=cropped_img, global_img_noise_type=global_img_noise_type)
                cropped_window_l.append(cropped_noisy_img)
                if(len(cropped_window_l) >= sequence_len):
                    break
            cropped_window_np = np.array(cropped_window_l)

        elif(movement_type == 'TILT_DOWN'):
            # bounding box
            #bb_pos_x = 10
            #bb_pos_y = 10 
            #bb_w = 300
            #bb_h = 240

            # sliding window in x direction  --> TILT DOWN
            cropped_window_l = []
            for i in range(0, img.shape[0], step_size):
                if(bb_pos_y + noise_y[i] + i + bb_h >= img.shape[0]):
                    break

                img_with_rect = img.copy()
                pt1 = (bb_pos_x + noise_x[i], bb_pos_y + noise_y[i] + i)
                pt2 = (bb_pos_x + noise_x[i] + bb_w, bb_pos_y + noise_y[i] + i + bb_h)
                cropped_img = img[(bb_pos_y + noise_y[i] + i):(bb_pos_y + noise_y[i] + i + bb_h), (bb_pos_x + noise_x[i]):(bb_pos_x + noise_x[i] + bb_w), :]
                cropped_noisy_img = self.createGlobalImageNoise(img=cropped_img, global_img_noise_type=global_img_noise_type)
                cropped_window_l.append(cropped_noisy_img)
                if(len(cropped_window_l) >= sequence_len):
                    break
            cropped_window_np = np.array(cropped_window_l)
        
        elif(movement_type == 'TILT_UP'):
            # bounding box
            #bb_pos_x = 10
            #bb_pos_y = 400 
            #bb_w = 300
            #bb_h = 240

            # sliding window in x direction  --> TILT UP
            cropped_window_l = []
            for i in range(0, img.shape[0], step_size):
                if(bb_pos_y + noise_y[i] - i <= 0):
                    break

                img_with_rect = img.copy()
                pt1 = (bb_pos_x + noise_x[i], bb_pos_y + noise_y[i] - i)
                pt2 = (bb_pos_x + noise_x[i] + bb_w, bb_pos_y + noise_y[i] - i + bb_h)
                cropped_img = img[(bb_pos_y + noise_y[i] - i):(bb_pos_y + noise_y[i] - i + bb_h), (bb_pos_x + noise_x[i]):(bb_pos_x + noise_x[i] + bb_w), :]
                cropped_noisy_img = self.createGlobalImageNoise(img=cropped_img, global_img_noise_type=global_img_noise_type)
                cropped_window_l.append(cropped_noisy_img)
                if(len(cropped_window_l) >= sequence_len):
                    break
            cropped_window_np = np.array(cropped_window_l)       

        elif(movement_type == 'NA'):
            # bounding box
            #bb_pos_x = 10
            #bb_pos_y = 400 
            #bb_w = 300
            #bb_h = 240

            # sliding window in x direction  --> TILT UP
            cropped_window_l = []
            for i in range(0, img.shape[0], step_size):
                if(bb_pos_y + noise_y[i] <= 0):
                    break

                img_with_rect = img.copy()
                pt1 = (bb_pos_x + noise_x[i], bb_pos_y + noise_y[i])
                pt2 = (bb_pos_x + noise_x[i] + bb_w, bb_pos_y + noise_y[i] + bb_h)
                cropped_img = img[(bb_pos_y + noise_y[i]):(bb_pos_y + noise_y[i] + bb_h), (bb_pos_x + noise_x[i]):(bb_pos_x + noise_x[i] + bb_w), :]
                cropped_noisy_img = self.createGlobalImageNoise(img=cropped_img, global_img_noise_type=global_img_noise_type)
                cropped_window_l.append(cropped_noisy_img)
                if(len(cropped_window_l) >= sequence_len):
                    break
            cropped_window_np = np.array(cropped_window_l)      

        elif(movement_type == None):
            print("ERROR: select valid movement type!")
            exit()

        return cropped_window_np

    def run(self):
        print("run data generator")

        file_list = os.listdir(self.image_path)

        bb_w = 256
        bb_h = 256
        margin = 10

        pan_flag = False
        tilt_flag = False

        for i, file in enumerate(file_list):
            print("process frame (" + str(i+1) + "|" + str(len(file_list)) + "): " + str(self.image_path + file))
            img = cv2.imread(self.image_path + file)

            if(img.shape[0] <= bb_w or img.shape[1] <= bb_h):
                print("WARNING: image is too small! skip!")
                continue
            
            # CREATE PANS
            class_name = "pan"

            movement_type_l = ['PAN_RIGHT', 'PAN_LEFT']
            if(pan_flag == True): 
                movement_type = movement_type_l[0]
                pan_flag = False

                bb_pos_x = margin
                bb_pos_y = margin 
            else:
                movement_type = movement_type_l[1]
                pan_flag = True

                bb_pos_x = img.shape[1] - bb_w - margin
                bb_pos_y = margin

            img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type=movement_type,
                                         shaky_mode_type="STANDARD",
                                         global_img_noise_type="STANDARD",
                                         magnitude_percentage=50,
                                         sequence_len=50)
            #self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="PAN_RIGHT")
            self.saveSequenceAsVideo(sid=i+1, class_name=class_name, frames_np=img_seq_np, dim=(512, 512))

            
            # CREATE TILTS
            class_name = "tilt"

            movement_type_l = ['TILT_UP', 'TILT_DOWN']
            if(tilt_flag == True): 
                movement_type = movement_type_l[0]
                tilt_flag = False
                bb_pos_x = margin
                bb_pos_y = img.shape[0] - bb_h - margin
            else:
                movement_type = movement_type_l[1]
                tilt_flag = True

                bb_pos_x = margin
                bb_pos_y = margin 

            img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type=movement_type,
                                         shaky_mode_type="STANDARD",
                                         global_img_noise_type="STANDARD",
                                         magnitude_percentage=50,
                                         sequence_len=50)
            #self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="PAN_RIGHT")
            self.saveSequenceAsVideo(sid=i+1, class_name=class_name, frames_np=img_seq_np, dim=(512, 512))

           
            # CREATE TILTS
            class_name = "na"

            bb_pos_x = int(img.shape[1] / 2) - int(bb_w / 2)
            bb_pos_y = int(img.shape[0] / 2) - int(bb_h / 2)
            img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type='NA',
                                         shaky_mode_type="STANDARD",
                                         global_img_noise_type="STANDARD",
                                         magnitude_percentage=50,
                                         sequence_len=50)
            #self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="PAN_RIGHT")
            self.saveSequenceAsVideo(sid=i+1, class_name=class_name, frames_np=img_seq_np, dim=(512, 512))


       
        

    def runDemo(self):
        print("run demo data generator")

        img = cv2.imread(self.image_path)
        print(img.shape)

        ''''''
        bb_w = 256
        bb_h = 256
        margin = 10
        #bb_pos_x = int(img.shape[1] / 2) - int(bb_w / 2)
        #bb_pos_y = int(img.shape[0] / 2) - int(bb_h / 2)
 
        # bounding box
        bb_pos_x = margin
        bb_pos_y = margin 
        img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type='PAN_RIGHT',
                                         shaky_mode_type="STRONG",
                                         global_img_noise_type="STRONG",
                                         magnitude_percentage=100,
                                         sequence_len=50)
        print(img_seq_np.shape)
        self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="PAN_RIGHT")
        

        ''''''
        # bounding box
        bb_pos_x = img.shape[1] - bb_w - margin
        bb_pos_y = margin
        img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type='PAN_LEFT',
                                         shaky_mode_type="STRONG",
                                         global_img_noise_type="STRONG",
                                         magnitude_percentage=50,
                                         sequence_len=50)
        print(img_seq_np.shape)
        self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="PAN_LEFT")
        

        ''''''
        # bounding box
        bb_pos_x = margin
        bb_pos_y = img.shape[0] - bb_h - margin
        img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type='TILT_UP',
                                         shaky_mode_type="STRONG",
                                         global_img_noise_type="STRONG",
                                         magnitude_percentage=50,
                                         sequence_len=50)
        print(img_seq_np.shape)
        self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="TILT_UP")
        
        # bounding box
        bb_pos_x = margin
        bb_pos_y = margin 
        img_seq_np = self.createMovement(img=img,
                                         bb_w=bb_w, 
                                         bb_h=bb_h, 
                                         bb_pos_x=bb_pos_x, 
                                         bb_pos_y=bb_pos_y, 
                                         movement_type='TILT_DOWN',
                                         shaky_mode_type="STRONG",
                                         global_img_noise_type="STRONG",
                                         magnitude_percentage=50,
                                         sequence_len=50)
        print(img_seq_np.shape)
        self.visualizeSequence(frames_np=img_seq_np, dim=(512, 512), win_text="TILT_DOWN")
        
        

    def visualizeSequence(self, frames_np=None, dim=(960, 720), win_text="default"):
        for i in range(0, len(frames_np)):
            #cv2.rectangle(img_with_rect, pt1, pt2, (0, 255, 0), 1)
            #cv2.imshow("orig img", img_with_rect)
            #print(frames_np[i].shape)
            img_resized = cv2.resize(frames_np[i], dim)
            cv2.imshow(win_text + " - " + str(len(frames_np)), img_resized)
            k = cv2.waitKey(10)


    def saveSequenceAsVideo(self, sid=-1, class_name=None, frames_np=None, dim=(512, 512)):
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        class_path = self.results_path + "/" + str(class_name) + "/"
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        out = cv2.VideoWriter(class_path + str(class_name) + "_" + str(sid) + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 12, dim)
        for i in range(0, len(frames_np)):
            img_resized = cv2.resize(frames_np[i], dim)
            #cv2.imwrite(img_resized)
            out.write(img_resized)
        out.release()
        
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


image_path = "/data/share/datasets/cmc_v1/extracted_frames_train/"
results_path = "/data/share/datasets/cmc_v1/train/"
obj = CmcSynDataGenerator(image_path, results_path)
obj.run()
