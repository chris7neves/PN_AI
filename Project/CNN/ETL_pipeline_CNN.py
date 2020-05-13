#---------------------------------------------------------------------
# This file will deal with the import of the data, cleaning        #
# and manipulation of the heat map images, resolution changes,      #
# and preparation of the images before being used by the network.   #
# SQL fetches
#---------------------------------------------------------------------

# ---------- Imports ---------- #
from Utilities import *
import IO

import os
import numpy as np
import cv2
import torch
import torch.utils.data
import random
from sklearn.model_selection import train_test_split
# ----------------------------- #

width = 1
height = 1


print("Please make sure that the only folders found in the Image data directory")
print("are either A or T folders filled with their corresponding images.")

#img_folder_name = input("Enter the folder name containing the image folders: ")
img_folder_name = "Images"

cwd = os.getcwd()
#print("cwd: "+cwd)
#filename = os.path.basename(__file__)   # Takes the filename of the python script its running in
#directory = cwd.replace(filename, '')
directory = cwd.replace('\\CNN', '')
img_folder_path = os.path.join(directory, img_folder_name)
print("Looking in directory: " + img_folder_path)

def create_non_split(images_path):
    """Creates a non split set for split set validation purposes"""
    print("Storing images")
    data_feature_list = []
    data_label_list = []
    for root, dirs, files in os.walk(images_path, topdown=True):  # convert pic to array and append to list
        for directory in dirs:
            # print(directory)
            subdir_path = os.path.join(root, directory)
            if directory[0] == 'A':
                label = 1  # 1 is the label for ASD "A"
            elif directory[0] == 'T':
                label = 0  # 0 is the label for TYP "T"
            for _, _, subdir_files in os.walk(subdir_path, topdown=True):
                for file in subdir_files:
                    img_path = os.path.join(img_folder_path, directory, file)
                    # print("img_path: " + img_path)
                    data_feature_list.append(np.array(cv2.imread(img_path,0)))
                    data_label_list.append(label)

    data_feature_list_np = np.array(data_feature_list)
    data_label_list_np = np.array(data_label_list, dtype=np.uint8)

    print("Images stored as Numpy Arrays.")

    return data_feature_list_np, data_label_list_np

def create_train_test(images_path, to_tensor=False):
    print("Storing images")
    data_feature_list = []
    data_label_list = []
    for root, dirs, files in os.walk(images_path, topdown=True):  # convert pic to array and append to list
        for directory in dirs:
            subdir_path = os.path.join(root, directory)
            if directory[0] == 'A':
                label = 1  # 1 is the label for ASD "A"
            elif directory[0] == 'T':
                label = 0  # 0 is the label for TYP "T"
            else:
                # log the error here
                print("There was an error with" + str(directory)) # This is a placeholder print statement. This error should be logged
            for _, _, subdir_files in os.walk(subdir_path, topdown=True):
                for file in subdir_files:
                    img_path = os.path.join(img_folder_path, directory, file)
                    data_feature_list.append(np.array(cv2.imread(img_path,0)))
                    data_label_list.append(label)

    data_feature_list_np = np.array(data_feature_list)
    data_label_list_np = np.array(data_label_list, dtype=np.uint8)

    print("Images stored as Numpy Arrays.")

    feat_train, feat_test, label_train, label_test  = train_test_split(data_feature_list_np, data_label_list_np,
                                                                      test_size=0.3, stratify=data_label_list_np)

    if to_tensor:
        return torch.from_numpy(label_train), torch.from_numpy(
            label_test), torch.from_numpy(feat_train), torch.from_numpy(
            feat_test)
    else:
        return label_train, label_test, feat_train, feat_test


class HeatMapDST(torch.utils.data.Dataset):
    """Heat Map Dataset Class"""

    def __init__(self, data, labels, train=True): # Need to test
        if train:
            self.data_category = "Training Data"
            self.train = True
        else:
            self.data_category = "Validation Data"
            self.train = False
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self): # Need to test
        if len(self.data) != len(self.labels):
            print("Error with label or data length in HeatMapDST class.")
            return 0
        return len(self.data)

    def __getitem__(self, index): # Need to test
        label = self.labels[index]
        image = self.data[index]
        return label, image

    def get_labels(self):
        return self.labels

    def get_features(self):
        return self.data

    def show_image(self, index=0, rand=False): # Need to test
        if rand:
            random.seed()
            last = self.__len__()
            temp_im = self.data[random.randint(0, last)]
            cv2.imshow('Image', temp_im.numpy())
        else:
            lab, img = self.__getitem__(index)
            img_name = "Label: " + str(lab)
            print(img_name)
            cv2.imshow(img_name, img.numpy())
            cv2.waitKey()

def get_dataloader(dset, batch_sz):
    hm_dloader = torch.utils.data.dataloader(dset, batch_size = batch_sz)
    return hm_dloader

######### Data Initialization #########

nonsplit_data, nonsplit_label = create_non_split(img_folder_path)

split_train_label, split_test_label, split_train_data, split_test_data = create_train_test(img_folder_path, to_tensor=False)

######### Data Verification #########

ones = sum([i == 1 for i in nonsplit_label])
zeros = len(nonsplit_label) - ones
print("There are %d ones and %d zeros is the nonsplit set." % (ones, zeros))

split_ones_train_label = sum([i == 1 for i in split_train_label])
split_zeros_train_label = len(split_train_label) - split_ones_train_label

split_ones_test_label = sum([i == 1 for i in split_test_label])
split_zeros_test_label = len(split_test_label) - split_ones_test_label

print("There are %d ones and %d zeros is the split training set." % (split_ones_train_label, split_zeros_train_label))
print("There are %d ones and %d zeros is the split testing set." % (split_ones_test_label, split_zeros_test_label))

######### DataSet Creation #########

train_dset = HeatMapDST(split_train_data, split_train_label, train=True)
test_dset = HeatMapDST(split_test_data, split_test_label, train=False)

######### DataSet Method Testing #########

print("The training dataset length is: {}".format(train_dset.__len__()))
print("The testing dataset length is: {}".format(test_dset.__len__()))

im1_label, im1_img = train_dset.__getitem__(3)
img_name = "im1 Label: {}".format(im1_label)
#cv2.imshow(img_name, im1_img.numpy())
#cv2.waitKey()

######## Creating the DataLoader ########

train_loader = get_dataloader(train_dset, batch_sz=10)
test_loader = get_dataloader(test_dset, batch_sz=10)




