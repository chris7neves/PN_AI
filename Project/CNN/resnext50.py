#---------------------------------------------------------------------
# This file will allow the user to define different sets of         #
# hyperparameters to use for different network architectures.       #
# This file will not build the network, it will accept user defined #
# parameters and call NetworkFactory to build and train the desired #
# architectures. This file will also display the results and        #
# performance metrics of the various defined networks.              #
#---------------------------------------------------------------------

import matplotlib
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import ETL_pipeline_CNN as etl
import NetworkFactory as net

batch_sz = 10


######### Data Initialization #########

nonsplit_data, nonsplit_label = etl.create_non_split(etl.img_folder_path)

split_train_label, split_test_label, split_train_data, split_test_data = etl.create_train_test(etl.img_folder_path, to_tensor=False)

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

train_dset = etl.HeatMapDST(split_train_data, split_train_label, train=True)
test_dset = etl.HeatMapDST(split_test_data, split_test_label, train=False)

######### DataSet Method Testing #########

print("The training dataset length is: {}".format(train_dset.__len__()))
print("The testing dataset length is: {}".format(test_dset.__len__()))

im1_label, im1_img = train_dset.__getitem__(3)
img_name = "im1 Label: {}".format(im1_label)
#cv2.imshow(img_name, im1_img.numpy())
#cv2.waitKey()

print("Get rand image")
test_dset.show_image(rand=True)

print("Shape of test dset: {}".format(test_dset.data.shape))