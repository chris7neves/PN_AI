#---------------------------------------------------------------------
# This file will deal with the import of the data, cleaning        #
# and manipulation of the heat map images, resolution changes,      #
# and preparation of the images before being used by the network.   #
# SQL fetches
#---------------------------------------------------------------------

# ---------- Imports ---------- #
from Utilities import *

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

# Get the label for the images generated by the heat map generator by looking at the first letter in the directory name
# -look in images directory
# --Walk through all the subdirectories to create numpy arrays holding the pixel data created using opencv
# --depending on array format, unsqueeze the dim=1 and insert a 1 for ASD or a 0 for TYP
# ---Create a tensor containing these array-diagnosis pairs

print("Please make sure that the only folders found in the Image data directory")
print("are either A or T folders filled with their corresponding images.")

#img_folder_name = input("Enter the folder name containing the image folders: ")
img_folder_name = "Images"

cwd = os.getcwd()
filename = os.path.basename(__file__)   # Takes the filename of the python script its running in
directory = cwd.replace(filename, '')
img_folder_path = os.path.join(directory, img_folder_name)
print("Looking in directory: " + img_folder_path)


def create_train_test(images_path, to_tensor=False):
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

    label_train, label_test, feat_train, feat_test = train_test_split(data_label_list_np, data_feature_list_np,
                                                                      test_size=0.3, stratify=data_label_list_np)
    #TODO: Verify if the label and feature arrays line up so that the features have the correct label at the right index
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
        if len(self.data) != len(self.data):
            print("Error with label or data length in HeatMapDST class.")
            return 0
        return len(self.data)

    def __getitem__(self, index): # Need to test
        label = self.label[index]
        image = self.data[index]
        return label, image

    def show_image(self, index=0, rand=False): # Need to test
        if rand:
            random.seed()
            last = self.__len__()
            temp_im = self.data[random.randint(0, last)]
            cv2.imshow('Image', temp_im)
        else:
            lab, img = self.__getitem__(index)
            img_name = "Label: " + str(lab)
            print(img_name)
            cv2.imshow(img_name, img)
            cv2.waitKey()


label_train, label_test, feat_train, feat_test = create_train_test(img_folder_path, to_tensor=False) #TODO: Test to see if the function creates equally balances train and test sets




# The following lines are for debugging purposes

# print(data_np[0][1]) # Prints out the first image in the array
# np.savetxt("arrayOut.txt",data_np[0][1], fmt='%.3i') # Saves the array image to a text file for debugging
# cv2.imwrite('testimage.jpg', data_np[0][1]) # write the array to a grayscale image
# cv2.imshow('image', data_np[0][1]) # displays the image
# cv2.waitKey() # waits for user input before closing image

# Creating the Heat Map dataset
#hm_dataset = HeatMapDST(img_folder_path)
#hm_dataset.show_image()

# Verifying length of the full dataset
#print(hm_dataset.__len__())

# Defining training and validation set sizes.
#train_size = 0.8 * hm_dataset.__len__()
#val_size = 0.2 * hm_dataset.__len__()

# Creating the training and validation sets. NOTE: Transforms cannot be applied to split datasets
#TODO: Check to see if the split sets are balanced
#trainset, valset = torch.utils.data.random_split(hm_dataset, [train_size, val_size])

# Defining batch size for dataloader
batch_sz = 10

# Creating the dataloaders
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True)
#val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_sz, shuffle=True)


