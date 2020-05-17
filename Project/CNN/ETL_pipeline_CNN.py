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

print("Please make sure that the only folders found in the Image data directory")
print("are either A or T folders filled with their corresponding images.")

#img_folder_name = input("Enter the folder name containing the image folders: ")
img_folder_name = "Images" # TODO: make this more robust and prompt user to choose using tkinter

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
        self.data = torch.unsqueeze(torch.from_numpy(data), 1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        if len(self.data) != len(self.labels):
            print("Error with label or data length in HeatMapDST class.")
            return 0
        return len(self.data)

    def __getitem__(self, index): # Need to test
        label = self.labels[index]
        image = self.data[index][0]
        return label, image

    def get_labels(self):
        return self.labels

    def get_features(self):
        return self.data

    def show_image(self, index=0, rand=False): # Need to test
        if rand:
            random.seed()
            last = self.__len__() - 1
            lab, temp_im = self.__getitem__(random.randint(0, last))
            cv2.imshow('Label: {}'.format(lab), temp_im.numpy())
            cv2.waitKey()
        else:
            lab, img = self.__getitem__(index)
            img_name = "Label: " + str(lab)
            print(img_name)
            cv2.imshow(img_name, img.numpy())
            cv2.waitKey()

def get_dataloader(dset, batch_sz):
    hm_dloader = torch.utils.data.DataLoader(dset, batch_size=batch_sz)
    return hm_dloader






