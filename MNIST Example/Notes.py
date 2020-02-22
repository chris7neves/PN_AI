# CUDA can easily be used to place certain obejcts and processes on GPU
import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

t = torch.tensor([1,2,3])   # Regular tensor object
print(t)

t = t.cuda()    # Places tensor onto GPU
print(t) #  tensor([1, 2, 3], device='cuda:0')  cude:0 is the first index. Pytorch places it on the first GPU

# Moving data from CPU to GPU is computationally intensive, so we should only place parallel computations on GPU
# cuDNN is the library that does deep neural net training on gpus

#------ Determine object type
print("t type is: " + str(type(t)))

# Determine Tensor shape
t.shape

#------ Reshaping Tensors
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(t.shape)

t.reshape(1, 9) # 1 array along first axis, 9 numbers along 2nd axis
                # Multiplication of shape sizes should equal number of elements
print(t.reshape(1, 9).shape) # reshaping also does not modify the tensor. It provides a temp reshaped copy


#------ 3 Important Tensor Attributes are

t.dtype # Type of data in tensor. Operations between tensors need to happen between tensor of the same types
t.device # If the tensor is located on cpu or gpu. Tensor operations need to happen between tensors found on the same device
t.layout # Layout of tensor. How it is laid out in memory. Advanced option. We can leave alone for now. Google "Stride of an Array"


#------- Creating Tensors with data. 4 different ways.

data = np.array([1, 2, 3])
print("Numpy array datatype: " + str(type(data)))

t1 = torch.Tensor(data) # Uppercase Tensor is the tensor class constructor. It will initialize the tensor with a
                        # predetermined datatype.
print("This is t1: " + str(type(t1)))

t2 = torch.tensor(data) # lowercase tensor is a factory function that creates tensor objects. Tensors created with this
                        # method will have a data type that matches the input data
print("This is t2 type: " + str(type(t2)))

t3 = torch.as_tensor(data)
print("This is t3: " + str(type(t3)))

t4 = torch.from_numpy(data)
print("This is t4: " + str(type(t4)))


#------ Creating Tensors without data

t5 = torch.eye(2) # Argument specifies number of rows. This generates an identity tensor, or a tensor with 1 in the diagonal

t6 = torch.zeros(2, 2) # Arguments are the length of each axis. 0s in all elements

t7 = torch.ones(2,2) # Same thing but has 1s everywhere

t8 = torch.rand(2, 2) # Arguemnts are the lengths of each axis initialized to random values between 0 and 1

#------ Explicitly setting the data type for a tensor

t9 = torch.tensor([1,2,3], dtype=torch.float64) # The constructor lacks this functionality. Use factory when possible

#------ Memory Sharing vs Copying of Tensor build types

data = np.array([1,2,3])
print(data)

t1 = torch.Tensor(data) # copes data
t2 = torch.tensor(data) # copies data
t3 = torch.as_tensor(data) # shares data
t4 = torch.from_numpy(data) # shares data

data[0] = 0
data[1] = 0
data[2] = 0

print(t1)
print(t2)
print(t3)
print(t4)

# last 2 methods share memory for performance, first two will copy. Moving between numpy
# arrays and pytorch tensors can be fast if memory is shared. Same memory pointer is shared. Any changes done to underlying
# data will have an effect on both variables.
# However, if performance is not important, always use torch.tensor(). If performance is very important, use .as_tensor()

#------ Reshaping Tensors

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

print(t.size)
print(t.shape) # tensor shape encodes infor about axes, rank and indices. 3 x 4. Shape returns an array
print(len(t.shape)) #essentially gives dimension, or rank.
print(torch.tensor(t.shape).prod()) # this essentially gives the amount of items in the tensor
print(t.numel()) # better way to find the number of elements.

# we need to be mindful of # of elements if were reshaping a tensor. ixj must always be equal to .numel()

print(t.reshape(1, 12))
print(t.reshape(2,6))
print(t.reshape(1, -1)) # VERY IMPORTANT. the -1 tells the reshape function to figure out what that value should be based
                        # on the number of elements in the tensor.

# squeezing a tensor removes all the dimensions with a length of 1
print(t.reshape(1, 12))
print(t.reshape(1, 12).shape) # prints [1, 12]
print(t.reshape(1, 12).squeeze())
print(t.reshape(1, 12).squeeze().shape) # prints [12]

# Flattening a tensor removes all axes except for 1, containing all elements of the tensor.
# We MUST flatten a tensor when we move from convolutional layer to the neural network.

#------ Making a flatten function

def flatten(t): # t is a pytorch tensor
    t = t.reshape(1, -1) # As mentioned previously, the -1 makes the reshape function determine what the appropriate value would be
    t = t.squeeze()
    return t

# or, more simply, but make sure to test

def flatten(t):
    return t.reshape(-1)

# or

t.flatten()

#------ Concatenating Tensors

t1 = torch.tensor([
    [1, 2],
    [3, 4]
])

t2 = torch.tensor([
    [5, 6],
    [7, 8]
])

print(torch.cat((t1, t2), dim=0)) # Dim=0 gives us row wise concatenations. We will get this
                                    # [[1, 2],
                                    #  [3, 4],
                                    #  [5, 6],
                                    #  [7, 8]]

print(torch.cat((t1, t2), dim=1)) # Dim=1 gives us column concatenation. We will get this:
                                    # [[1, 2, 5, 6],
                                    #  [3, 4, 7, 8]]


#------- Flatten example using pretend images as batches to pass to a CNN

# Pretend that we have 3 4x4 images

t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])
t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])
t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

# remember that the tensor that needs to be passed to the CNN have the following format
# (batch size, channels, height, width)

#batches are represented using a single tensor, so we need to combine these 3 tensors into a single tensor that has 3 axes, instead of 2

t = torch.stack((t1, t2, t3)) # the stack method concatenates the vectors along a new axis and does not use an existing one
print(t.shape) # prints [3, 4, 4]

# So we now have 3 images of 4x4 pixels in the tensor 3. Only thing missing now is color channels.
# These images are assumed to be black and white so we need jsut one color channel.

t = t.reshape(3,1,4,4)

# were adding another axis. The number of elements doesnt change though, because the axes multiplied together will not change
# by multiplying them by 1
print(t[0]) # First image
print(t[0][0]) # first color channel for first image
print(t[0][0][0]) # First row of pixels of first image in first color channel
print(t[0][0][0][0]) # This gives the first pixel value in the first row of the first color channel of the first image

# now we want to flatten the images. Not the entire tensor t, as that would flatten the batches and combine all the pictures together,
# which we dont want. We want to flatten the (C,H,W) axes

t.flatten(start_dim=1) # The start_dim argument tells flatten() which axis to start the flatten at. We use 1 as that skips over
                        # index 0, which is the batch axis. So each image will flatten on itself but will still remain distinct

print(t.flatten(start_dim=1).shape)

# flattening an RGB image:


# ------ Element wise operations

# element wise operations need the same shape of tensors to be performed
# all arithmetic operations in pytorch are element wise

t1 = torch.tensor([
    [1,2],
    [3,4]
])
t2 = torch.tensor([
    [5,6],
    [7,8]
])

print(t1+t2) # the two tensors need to be of the same shape and are added element wise
print(t1+2) # scalars can be added to tensors and it performs an element wise addition
print(t1 * 2) # can multiply and divide by scalars too

# ------- Tensor Broadcasting

# we are able to add a tensor with a scalar, and do other operations with scalars and tensors of different shapes
# because pytorch is broadcasting values to the same shape as the tensor under the hood
# Broadcasting simply means that the scalar is extended to an array with the shape required to interact with the tensor
# We can also do boolear operations due to this feature.

t1 = torch.tensor([
    [1,2],
    [-1,4]
])

print(t1 > 0) # This will print an array with the boolean results for each comparison at each index

t1 = torch.tensor([
    [1,1],
    [1,1]
])

t2 = torch.tensor([
    [2,4]
])

print(t1 + t2) # This will print a tensor with [3, 5] in each row as a 2x2 matrix

print(np.broadcast_to(t2.numpy(), t1.shape)) # We can use this numpy function to help us get an idea of what the broadcasting
                                             # does to the tensor

t1 = torch.tensor([
    [1,1,1],
    [1,1,1]
])

# print(t1 + t2) # This will not work

# ------ Some built in comparison operators

t = torch.tensor([
    [0,5,0],
    [6,0,7],
    [0,8,0]
])

# For the comparison operators, 0 is FALSE and 1 is TRUE

print(t.eq(0)) # element wise check for equality to the argument
print(t.ge(0)) # element wise greater or equal to the argument
print(t.gt(0)) # Greater than
print(t.lt(0)) # Less than
print(t.le(7)) # Less than or equal to

# other element wise operations
t = t * 1.0
t.abs() # element wise absolute value
print(t.sqrt()) # element wise square root
t.neg() # element wise negative ( * -1)


#------ Tensor Reduction operations

t = torch.tensor([
    [0,1,0],
    [2,0,2],
    [0,3,0]
])

t = t * 1.0
print(t.sum())
print(t.prod())
print(t.mean())
print(t.std())

# We can even reduce along certain axes by passing a dimension argument to the function

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

print(t.sum(dim=0))
print(t.sum(dim=1))

#------ ArgMax
# argmax returns a tensor with the index value of the highest value in the tensor
# it flattens the tensor completely, then retruns the index of the max value

t = torch.tensor([
    [1,0,0,2], #index 0
    [0,3,3,0], #index 1
    [4,0,0,5]  #index 2
])

print(t.max())
print(t.argmax())
print(t.flatten()) # The 11th index of the flattened tensor is 5
print(t.max(dim=0)) # yields the max value in the "row" dimensions in the tensor. Performs element wise maximum on the first axis (so it compares (1, 0, 4), (0, 3, 0), (0, 3, 0), and (2, 0, 5)
print(t.argmax(dim=0))

# we typically use the argmax function on the NN output prediction tensor

#------ Item tensor method

t = t * 1.0
print(t.mean())
print(t.mean().item())

t = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

# we can find values of certain dimensions and export that dimension to a Python list or Numpy array
print(t.mean(dim=0).tolist())
print(t.mean(dim=0).numpy())


#------------------- PYTORCH NEURAL NET CLASSES AND METHODS --------------------------------

# torch.utils.data.Dataset # abstract class for representing a dataset
# torch.utils.data.DataLoader # wraps a dataset and provides access to the underlying data

# to create a custom dataset using pytorch, we need to extend the Dataset class by creating a subclass that implements
# the required methods
# Once this is done, the created custom dataset can be passed to the DataLoader object

# an example custom Dataset subclass would look like this

class OHLC(torch.utils.data.Dataset):
    def __init(self, csv_file):
        self.data = pd.read_csv(csv_file) # Reads a csv into a pandas dataframe

    def __getitem__(self, index): # This method is necessary if a custom dataset is built
        r = self.data.iloc[index] # iloc is a pandas purely integer location based indexing method that returns the item at the particular position in the pandas dataframe. Can return a series or another dataframe depending on date
        label = torch.tensor(r.is_up_day, dtype=torch.long)
        sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
        return sample, label

    def __len__(self): # returns dataset length
        return len(self.data)

# most basic network

class Network:
    def __init__(self):
        self.layer = None

    def forward(self, t): # forward takes in a tensor t, applies a transform, then returns it
        t = self.layer(t)
        return t

# Pytorch neural net

class Network(nn.Module): # Module baseclass keeps track of networks weights that are contained within each layer
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t): # forward takes in a tensor t, applies a transform, then returns it
        t = self.layer(t)
        return t