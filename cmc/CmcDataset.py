import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from PIL import Image
import torch.nn as nn  # Add on classifier
import os
import cv2

# pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# pip install Pillow
# pip install matplotlib
# custom opencv
# pip install h5py

#######################
## define network
#######################


class CmcDataset(data.Dataset):

    def __init__(self, path=None, db_set=None, shuffle=True, subset=None, transform=None, target_transform=None):
        'Initialization'
        self.path = path
        self.shuffle = shuffle
        self.subset = subset  # [600, 120, 120]
        self.db_set = db_set
        self.transform = transform
        self.target_transform = target_transform

        annotation_path = self.path + "annotation/"

        if(db_set == "train"):
            #annotation_file = annotation_path + "/train_shots.flist"
            annotation_file = annotation_path + "/train_shots.csv"
        elif(db_set == "val"):
            #annotation_file = annotation_path + "/val_shots.flist"
            annotation_file = annotation_path + "/val_shots.csv"
        elif(db_set == "test"):
            #annotation_file = annotation_path + "/test_shots.flist"
            annotation_file = annotation_path + "/test_shots.csv"
        else:
            print("select valid dataset type!")
            print(db_set)
            exit()

        classes = ["pan", "tilt", "track"]

        fp = open(annotation_file, "r")
        lines = fp.readlines()
        fp.close()

        annotations_l = []
        for line in lines:
            line = line.replace("\n", "")
            #line_split = line.split(" ")
            line_split = line.split(";")
            #print(int(line_split[1]))
            #print(line)
            annotations_l.append([self.path + line_split[0], classes[int(line_split[1])]])
        annotations_np = np.array(annotations_l)

        #print(annotations_np)
        #print(len(annotations_np))
        self.samples = annotations_np

        if (self.shuffle == True):
            self.shuffled_ids = np.arange(len(self.samples))
            np.random.shuffle(self.shuffled_ids)
            np.random.shuffle(self.shuffled_ids)
            np.random.shuffle(self.shuffled_ids)
        else:
            self.shuffled_ids = np.arange(len(self.samples))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        # LOAD VIDEO
        vid_path = self.samples[self.shuffled_ids[index]][0]
        label = self.samples[self.shuffled_ids[index]][1]
        ''' '''

        classes = {"pan": 0, "tilt": 1, "track": 2}
        class_names = ["pan", "tilt", "track"]

        shot_frames_l = []
        vid_instance = cv2.VideoCapture(vid_path)

        # read first frame
        ret, frame = vid_instance.read()

        if(ret == True):
            shot_frames_l.append(frame)

            num_frames = int(vid_instance.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(num_frames)
            height = int(vid_instance.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(vid_instance.get(cv2.CAP_PROP_FRAME_WIDTH))
            channels = 3
        else:
            #print("----")
            #print(ret)
            #print(vid_path)
            height = 720
            width = 960
            channels = 3
            num_frames = 1
            frame = np.zeros([height, width, channels], dtype=np.uint8)
            shot_frames_l.append(frame)
            #print(frame.shape)


        num_seq = 32
        stride = 1
        for i in range(1, num_seq*stride, stride):
            #print("A")
            if(i >= num_frames):
                frame = np.zeros([height, width, channels], dtype=np.uint8)
                #frame_gray = np.zeros([height, width], dtype=np.uint8)
                #print("padding")
                #print(frame.shape)
                #print(frame.dtype)
                shot_frames_l.append(frame)
            else:
                vid_instance.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = vid_instance.read()
                if (ret == True):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #frame_pil = Image.fromarray(frame)
                    shot_frames_l.append(frame)

        shot_frames_np = np.array(shot_frames_l)
        vid_instance.release()
        #print(shot_frames_np.shape)
        #exit()

        if (self.transform):
            seq_tensor = self.transform(shot_frames_np)
        else:
            print("ERROR: you have to specify a valid transformation object!")
            exit()

        '''
        #transforms.Resize((720, 960)),
        transforms.CenterCrop((720, 720)),
        transforms.Resize((128, 128)),
        #HorizontalFlip3D(),
        #VerticalFlip3D(),
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.RandomVerticalFlip(),  # randomly flip and rotate
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize((94.05657 / 255.0, 94.05657 / 255.0, 94.05657 / 255.0),
                             (57.99793 / 255.0, 57.99793 / 255.0, 57.99793 / 255.0))
        '''
        #shot_frames_np = np.array(shot_frames_l)
        #seq = torch.stack(shot_frames_l)

        return seq_tensor, int(classes[label])
