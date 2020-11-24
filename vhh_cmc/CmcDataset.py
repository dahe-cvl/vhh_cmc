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

        if (db_set == None):
            print("select valid dataset type!")
            print(db_set)
            exit()

        # load all samples
        class_names_l = os.listdir(self.path + self.db_set)

        all_samples_l = []
        for i, class_name in enumerate(class_names_l):
            samples_l = os.listdir(self.path + "/" + self.db_set + "/" + class_name)
            for sample in samples_l:
                all_samples_l.append([self.path + "/" + self.db_set + "/" + class_name + "/" + sample, i])
        self.samples = np.array(all_samples_l)
        self.classes = class_names_l

        #print(self.samples)
        #print(self.classes)
        #exit()

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

        label_tensor = torch.tensor(int(label))

        frames_l = []
        vid_instance = cv2.VideoCapture(vid_path)
        ret, frame = vid_instance.read()

        seq_len = 50
        frame_dim = frame.shape

        sequence_np = np.zeros((seq_len, frame_dim[0], frame_dim[1], frame_dim[2])).astype('uint8')

        sequence_np[0] = frame
        for i in range(1, len(sequence_np)):
            ret, frame = vid_instance.read()
            if(ret == True):
                sequence_np[i] = frame       
            
        vid_instance.release()

        if (self.transform):
            seq_tensor = self.transform(sequence_np)
        else:
            print("ERROR: you have to specify a valid transformation object!")
            exit()
        #print(seq_tensor.size())
        #print(label_tensor)
        #exit()
        return seq_tensor, label_tensor
        