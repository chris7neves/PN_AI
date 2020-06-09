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
import Utilities

batch_sz = 10


######### Data Initialization #########

nonsplit_data, nonsplit_label = etl.create_non_split(etl.img_folder_path, balance=False)

print(f"Nonsplit Dataset Shape: {nonsplit_data.shape}")
print(f"Nonsplit Label Shape: {nonsplit_label.shape}")

split_train_label, split_test_label, split_train_data, split_test_data = etl.split_train_test(nonsplit_data
                                                                                              , nonsplit_label
                                                                                              , to_tensor=False)

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

######### Creating Cuda Device to GPU #########

device = torch.device('cuda:0')

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

#print("Get rand image")
#test_dset.show_image(rand=True)

print("Shape of test dset: {}".format(test_dset.data.shape))

########## Training the CNN ##########

torch.set_grad_enabled(True)

BasicCNN = net.basicCNN().to(device)


optimizer = optim.Adam(BasicCNN.parameters(), lr=0.001)
batch_size = 20

train_loader = etl.get_dataloader(train_dset, batch_sz=batch_size, shuff=True)
test_loader = etl.get_dataloader(test_dset, batch_sz=batch_size, shuff=True)

count = 0
print("Starting training...")
for epoch in range(10): #TODO: write function to automatically go to next epoch once progress of current epoch plateaus
    total_loss = 0
    total_correct = 0
    total_guessed = 0
    for i, batch in enumerate(train_loader):
        print("Epoch:{}".format(epoch), "batch:{}".format(i), 'START')
        images, labels = batch
        img_gpu = images.to(device)
        labels_gpu = labels.to(device)

        print(images.shape)
        #predictions = BasicCNN(images)
        predictions = BasicCNN(img_gpu)

        print("Calculating loss...")
        #loss = F.cross_entropy(predictions, labels.long()) # Calculates the loss function using cross entropy
        loss = F.cross_entropy(predictions, labels_gpu.long())

        print(f"Loss: {loss}")
        print("Resetting gradients...")
        optimizer.zero_grad() # Resets gradients from previous cycle

        print("Using backpropagation for gradient calculation...")
        loss.backward() # Uses backprop to calc gradients

        print("Updating weights...")
        optimizer.step() # Updates the weights

        total_loss += loss.item()
        #total_correct += Utilities.get_num_correct(predictions, labels)
        total_correct += Utilities.get_num_correct(predictions, labels_gpu)

        total_guessed += images.shape[0]


        print(f"Epoch: {epoch} batch: {i}  accuracy: {total_correct/total_guessed} COMPLETE\n")

    print("Epoch:{}".format(epoch), "Total loss:{}\n\n".format(total_loss))

with torch.no_grad():
    all_preds = torch.tensor([], dtype=torch.uint8).to(device)
    all_labels = torch.tensor([], dtype=torch.uint8).to(device)
    for batch in test_loader:
        test_images, test_labels = batch
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        preds = BasicCNN(test_images)
        all_preds = torch.cat((all_preds, preds.type(torch.uint8)), dim=0)
        all_labels = torch.cat((all_labels, test_labels.type(torch.uint8)), dim=0)
    tot_correct = Utilities.get_num_correct(all_preds, all_labels)
    accuracy = tot_correct/(all_labels.shape[0])
    print(f"Total correct: {tot_correct}  out of {(all_labels.shape[0])} guesses.")
    print(f"End accuracy: {accuracy}")