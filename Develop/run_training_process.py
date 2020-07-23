from cmc.CmcDataset import CmcDataset
from cmc.Model import CNNModel, resnet50, CNN3D, CnnLstm
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import h5py as hf
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn  # Add on classifier
import os
from torch.autograd import Variable
import cv2
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

print("start training ... ")


def crop(img: np.ndarray, dim: tuple):
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

class CenterCrop3D(object):
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim

    def __call__(self, frame_seq):
        final_frames = []
        for i in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[i])
            frame_cropped = crop(frame, self.crop_dim)
            final_frames.append(frame_cropped)
        final_frames_np = np.array(final_frames)
        return final_frames_np

    def __repr__(self):
        return self.__class__.__name__ + '_Resize3D_'

class ToGrayScale3D(object):
    def __call__(self, frame_seq):
        #print(frame_seq.shape)

        final_frames = []
        for i in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[i])
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            final_frames.append(np.expand_dims(frame_gray, axis=2))
        final_frames_np = np.array(final_frames)

        #print(final_frames_np.shape)
        #exit()
        return final_frames_np

    def __repr__(self):
        return self.__class__.__name__ + '_ToGrayScale3D_'

class Resize3D(object):
    def __init__(self, resize_dim):
        self.resize_dim = resize_dim

    def __call__(self, frame_seq):
        #print(frame_seq.shape)
        final_frames = []
        for i in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[i])
            frame_resized = cv2.resize(frame, (self.resize_dim[1], self.resize_dim[0]))
            final_frames.append(frame_resized)
        final_frames_np = np.array(final_frames)

        return final_frames_np

    def __repr__(self):
        return self.__class__.__name__ + '_Resize3D_'


class HorizontalFlip3D(object):

    def __call__(self, frame_seq):
        opt = random.randint(0, 1)

        final_frames = []
        for i in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[i])
            if (opt == 1):
                frame = cv2.flip(frame, flipCode=1)
            final_frames.append(frame)

        final_frames_np = np.array(final_frames)
        return final_frames_np

    def __repr__(self):
        return self.__class__.__name__ + '_horizontal_flip_3D_'

class VerticalFlip3D(object):

    def __call__(self, frame_seq):
        opt = random.randint(0, 1)

        final_frames = []
        for i in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[i])
            if(opt == 1):
                frame = cv2.flip(frame, flipCode=0)
            final_frames.append(frame)

        final_frames_np = np.array(final_frames)
        return final_frames_np

    def __repr__(self):
        return self.__class__.__name__ + '_vertical_flip_3D_'


class RandomRotate3D(object):
    def __init__(self, p=0.5, max_rot_range=None):
        self.p = p
        self.max_rot_range = max_rot_range

    def __call__(self, frame_seq):
        angle = random.randint(self.max_rot_range[0], self.max_rot_range[1])

        final_frames = []
        for i in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[i])
            image_center = tuple(np.array(frame.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
            final_frames.append(frame)

        final_frames_np = np.array(final_frames)

        return final_frames_np

    def __repr__(self):
        return self.__class__.__name__ + '__RandomRotate3D_'

class ToTensorShape3D(object):

    def __call__(self, frame_seq):

        frame_seq_reshaped = np.expand_dims(frame_seq, axis=0)
        frame_seq_reshaped = frame_seq_reshaped.transpose((1,0,4,2,3))
        #print(frame_seq.shape)
        #print(np.expand_dims(frame_seq, axis=0).shape)
        #print(np.expand_dims(frame_seq, axis=0).transpose((1,0,4,2,3)).shape)
        #exit()


        return frame_seq_reshaped

    def __repr__(self):
        return self.__class__.__name__ + '_ToTensorShape3D_'

class ToNormalizedTensor3D(object):

    def __call__(self, frame_seq):
        #print(frame_seq.shape)

        shot_frames_l = []
        trans1 = transforms.ToTensor()
        trans2 = transforms.Normalize((106.79066460039768 / 255.0,
                                       112.39727916518281 / 255.0,
                                       113.79938372074331 / 255.0
                                      ),
                                      (75.09153312342316 / 255.0,
                                       73.49875024709992 / 255.0,
                                       73.11534896091715 / 255.0
                                      ))

        for a in range(0, len(frame_seq)):
            frame = np.asarray(frame_seq[a])
            shot_frames_l.append(trans2(trans1(frame)))  #trans2
        seq = torch.stack(shot_frames_l)

        #print(seq.size())
        #print(seq.reshape((seq.size()[1], seq.size()[0], seq.size()[2], seq.size()[3])).size())
        #exit()
        # CNN-LSTM
        seq_final = seq.reshape((seq.size()[0], seq.size()[1], seq.size()[2], seq.size()[3]))
        # 3DCNN
        #seq_final = seq.reshape((seq.size()[1], seq.size()[0], seq.size()[2], seq.size()[3]))
        #print(seq_final.size())
        #exit()
        return seq_final

    def __repr__(self):
        return self.__class__.__name__ + '_horizontal_flip_3D_'

def loadDataset(path="", batch_size=64):
    if (path == "" or path == None):
        print("ERROR: you must specifiy a valid dataset path!")
        exit()

    transform_train = transforms.Compose([
        Resize3D((720, 960)),
        CenterCrop3D((720, 720)),
        Resize3D((128, 128)),
        #ToGrayScale3D(),
        HorizontalFlip3D(),
        VerticalFlip3D(),
        RandomRotate3D(max_rot_range=[-15, 15]),
        #ToTensorShape3D(),
        ToNormalizedTensor3D(),
    ])

    transform_val = transforms.Compose([
        Resize3D((720, 960)),
        CenterCrop3D((720, 720)),
        Resize3D((128, 128)),
        #ToGrayScale3D(),
        ToNormalizedTensor3D(),
    ])

    transform_test = transforms.Compose([
        Resize3D((720, 960)),
        CenterCrop3D((720, 720)),
        Resize3D((256, 256)),
        #ToGrayScale3D(),
        ToNormalizedTensor3D(),
    ])

    db_path = path
    train_data = CmcDataset(path=db_path,
                            db_set="train",
                            shuffle=True,
                            subset=None,
                            transform=transform_train,
                            target_transform=None)

    valid_data = CmcDataset(path=db_path,
                            db_set="val",
                            shuffle=False,
                            subset=None,
                            transform=transform_val,
                            target_transform=None)

    test_data = CmcDataset(path=db_path,
                            db_set="test",
                            shuffle=False,
                            subset=None,
                            transform=transform_test,
                            target_transform=None)

    num_workers = 6
    # Dataloader iterators, make sure to shuffle
    trainloader = DataLoader(train_data,
                             batch_size=batch_size,
                             # sampler=train_sampler,
                             shuffle=True,
                             num_workers=num_workers
                             )

    # print(np.array(trainloader.dataset).shape)

    validloader = DataLoader(valid_data,
                             batch_size=batch_size,
                             # sampler=valid_sampler,
                             shuffle=False,
                             num_workers=num_workers
                             )

    testloader = DataLoader(test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers
                            )

    print("train samples: " + str(len(train_data)))
    print("valid samples: " + str(len(valid_data)))
    print("test samples: " + str(len(test_data)))

    return trainloader, validloader, testloader

def matplotlib_imshow(img, one_channel=False):
    print("imshow function ... ")
    print(img.size())

    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def calculate_statistics(trainloader):
    all_frames_l = []
    for i, (inputs, labels) in enumerate(trainloader):
        #print("----")
        #print(i)
        #print(inputs.size())
        #print(labels.size())
        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        print(inputs.size())
        print(type(inputs))

        inputs_np = inputs.detach().cpu().numpy()
        print(inputs_np.shape)
        all_frames_l.extend(inputs_np)
        print(np.min(inputs_np))
        print(np.max(inputs_np))
        #exit()

    all_frames_np = np.array(all_frames_l)
    print(all_frames_np.shape)

    all_frames_np = np.reshape(all_frames_np, (all_frames_np.shape[0] * all_frames_np.shape[1],
                                               all_frames_np.shape[2],
                                               all_frames_np.shape[3],
                                               all_frames_np.shape[4]))

    print(all_frames_np.shape)
    all_samples_r_np = all_frames_np[:, :1, :, :]
    all_samples_g_np = all_frames_np[:, 1:2, :, :]
    all_samples_b_np = all_frames_np[:, 2:3, :, :]

    print("calculate mean value for each color channel ... ")
    mean_r = np.mean(all_samples_r_np)
    mean_g = np.mean(all_samples_g_np)
    mean_b = np.mean(all_samples_b_np)
    print(mean_r)
    print(mean_g)
    print(mean_b)

    print("calculate standard deviation of zero-centered frames ... ")
    std_r = np.std(all_samples_r_np)
    std_g = np.std(all_samples_g_np)
    std_b = np.std(all_samples_b_np)
    print(std_r)
    print(std_g)
    print(std_b)

    # 106.79066460039768 / 255.0
    # 112.39727916518281 / 255.0
    # 113.79938372074331 / 255.0
    # 75.09153312342316 / 255.0
    # 73.49875024709992 / 255.0
    # 73.11534896091715 / 255.0

db_path = "/data/share/cmc_final_dataset_v2/"
#db_path = "/data/share/cmc_eval_dataset/"
n_epochs = 100
expFolder = "test_exp_15"
early_stopping_threshold = 30
wDecay = 0.0
lr = 0.001
batch_size = 64

trainloader, validloader, testloader = loadDataset(path=db_path, batch_size=batch_size)
#calculate_statistics(trainloader)
#exit()

train_on_gpu = torch.cuda.is_available()
print("Train on gpu: " + str(train_on_gpu))

# Number of gpus
multi_gpu = False
if train_on_gpu:
    gpu_count = torch.cuda.device_count()
    print("gpu_count: " + str(gpu_count))
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

################
# define model
################
#model = CNN3D(t_dim=32)
#model = CNNModel()
model = CnnLstm()

print(model)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print("total_params:" + str(total_params))
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print("total_trainable_params: " + str(total_trainable_params))

if train_on_gpu:
    model = model.to('cuda')

if multi_gpu:
    model = nn.DataParallel(model)


#exit()

################
# Specify the Loss function
################
criterion = nn.CrossEntropyLoss()

################
# Specify the optimizer
################
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wDecay)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wDecay)

# print("[Creating Learning rate scheduler...]")
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

# Define the lists to store the results of loss and accuracy
best_acc = 0.0
best_loss = 10.0
early_stopping_cnt = 0





'''
writer = SummaryWriter(log_dir="./runs/" + expFolder)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.size())
print(images.reshape(images.size()[0], images.size()[2], images.size()[1], images.size()[3], images.size()[4]).size())
images = images.reshape(images.size()[0], images.size()[2], images.size()[1], images.size()[3], images.size()[4])
print(images.size())

# create grid of images

for a in range(0, 5):

    img_grid = torchvision.utils.make_grid(images[a])
    print(img_grid.size())

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('test_' + str(a), img_grid)

writer.close()
exit()
'''

writer = SummaryWriter(log_dir="./runs/" + expFolder)

for epoch in range(0, n_epochs):
    tLoss_sum = 0
    tAcc_sum = 0
    vLoss_sum = 0
    vAcc_sum = 0
    ###################
    # train the model #
    ###################
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        #print("----")
        #print(i)
        #print(inputs.size())
        #print(labels.size())
        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        # If we have GPU, shift the data to GPU
        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # run forward pass
        outputs = model(inputs)
        tLoss = criterion(outputs, labels)
        tLoss_sum += tLoss.item()

        # run backward pass
        optimizer.zero_grad()
        tLoss.backward()
        optimizer.step()

        preds = outputs.argmax(1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum()
        acc = correct.float() / preds.shape[0]
        tAcc_sum += acc.item()

    ###################
    # train the model #
    ###################
    model.eval()
    for i, (inputs, labels) in enumerate(validloader):
        # print("----")
        # print(i)
        # print(inputs.size())
        # print(labels.size())
        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        # If we have GPU, shift the data to GPU
        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # run forward pass
        outputs = model(inputs)

        # print(outputs)
        # exit()
        vLoss = criterion(outputs, labels)
        vLoss_sum += vLoss.item()

        preds = outputs.argmax(1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum()
        acc = correct.float() / preds.shape[0]
        vAcc_sum += acc.item()

    print('Epoch [{:d}/{:d}]: train_loss: {:.3f}  train_acc: {:.3f} val_loss: {:.3f} val_acc: {:.3f}'.format(
        epoch + 1, n_epochs, tLoss_sum / len(trainloader), tAcc_sum / len(trainloader), vLoss_sum / len(validloader), vAcc_sum / len(validloader)))

    ###############################
    # write results to tensorboard
    ###############################
    writer.add_scalar('train_loss', tLoss_sum / len(trainloader), epoch)
    writer.add_scalar('valid_loss', vLoss_sum / len(validloader), epoch)
    writer.add_scalar('train_acc', tAcc_sum / len(trainloader), epoch)
    writer.add_scalar('valid_acc', vAcc_sum / len(validloader), epoch)

    ###############################
    # Save checkpoint.
    ###############################
    acc_curr = 100. * (vAcc_sum / len(validloader))
    vloss_curr = vLoss_sum / len(validloader)
    if acc_curr > best_acc:
        print('Saving...')
        state = {
            'net': model.state_dict(),
            'acc': acc_curr,
            'loss': vloss_curr,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        torch.save(state, "/home/dhelm/VHH_Develop/pycharm_vhh_cmc/runs/" +
                   expFolder + "/" "best_model" + ".pth")  # + str(round(acc_curr, 4)) + "_" + str(round(vloss_curr, 4))
        best_acc = acc_curr
        # best_loss = vloss_cur
        early_stopping_cnt = 0
    # scheduler.step()

    ###############################
    # early stopping.
    ###############################
    if (acc_curr <= best_acc):
        early_stopping_cnt = early_stopping_cnt + 1
    if (early_stopping_cnt >= early_stopping_threshold):
        print('Early stopping active --> stop training ...')
        break
''''''
writer.close()

###################
# test the model #
###################

model.load_state_dict(torch.load("/home/dhelm/VHH_Develop/pycharm_vhh_cmc/runs/" +
                   expFolder + "/best_model.pth")['net'])

vLoss_sum = 0
vAcc_sum = 0
model.eval()
for i, (inputs, labels) in enumerate(testloader):
    # print("----")
    # print(i)
    # print(inputs.size())
    # print(labels.size())
    # Convert torch tensor to Variable
    inputs = Variable(inputs)
    labels = Variable(labels)

    # If we have GPU, shift the data to GPU
    CUDA = torch.cuda.is_available()
    if CUDA:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # run forward pass
    outputs = model(inputs)
    # print(outputs)
    # exit()
    vLoss = criterion(outputs, labels)
    vLoss_sum += vLoss.item()

    preds = outputs.argmax(1, keepdim=True)
    correct = preds.eq(labels.view_as(preds)).sum()
    acc = correct.float() / preds.shape[0]
    vAcc_sum += acc.item()

print("test loss: " + str(vLoss_sum / len(testloader)))
print("test accuracy: " + str(vAcc_sum / len(testloader)))
''''''


'''
print(db_instance.__len__())
sample, label = db_instance.__getitem__(0)
print(sample.shape)
print(label)
'''


