import torch
import numpy as np
import matplotlib
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim # library that contains optimization functions
import pandas as pd
import itertools # used for confusion matrix

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(False) # Turn this on when training. This modifies the weights by calculating the partial deriv after each calc
# 1 - Data Import and Preprocessing (we will need to make sure that the PN_AI dataset is balanced)

# Training set of fashionMNIST

# download arg downloads the data set if it is not already found in root
# transform arg transforms the images into tensors using torchvision.transforms
train_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST', train=True, download=True
                                              , transform= transforms.Compose([transforms.ToTensor()]) )

# Wrapping it with the dataloader object

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10) # The dataloader wrapper provides useful utilities like shuffling, batch size and others
# pytorch documentation specifies that the first axis is the batch axis. Batching takes multiple data points and collates them into a batch

print(len(train_set))

print(train_set.targets) # The labels represent the class that the data point is in. 9 is an ankleboot, 0 is a t shirt.

print(train_set.targets.bincount()) # a bin is technically a folder. We call bincount on the train_labels and it creates a bin for each class
                                         # and tells us how many data points are in each of the classes. If classes have a varying number of samples, then
                                         # we say that the dataset is imbalanced.
                                         # Your training and validation sets need to have a good representation of what data is
                                         # in the legitimate set
                                         # If your training set is unbalanced and has more of one class than the other, you
                                         # can replicate the classes that dont have as much data so that you have the same
                                         # amount for every class

# Structure of the MNIST data:

sample = next(iter(train_set)) # we take the training set and make it into an iterable and we use next to navigate

print(len(sample)) # the sample has a length of 2
print(type(sample)) # and is a tuple
image, label = sample # the sample is a tuple containing both the label and the actual image itself

print(image.shape)
# print(label.shape) This will print an error as label is an integer and has no "shape" attribute

plt.imshow(image.squeeze(), cmap='gray') # squeeze gets rid of the empty color channel dimesion as it is 1 for grayscale
print('label: ', label)
plt.show()
# imshow is a matplotlib function that plots an Array image to a 2D mapping with the specified color map
# the squeeze method is used to ensure that only the axes containing color values are kept and any single dimensional entries are removed

batch = next(iter(train_loader)) # The batch size is defined when calling the data_loader wrapper

print(len(batch))

print(type(batch))

images, labels = batch

print(images.shape)
print(labels.shape)

grid = torchvision.utils.make_grid(images, nrow=10) # nrow is the amount of images in each row

plt.imshow(np.transpose(grid, (1,2,0))) # we need to make the pytorch array (channel (0), height (1), width (2)) formated to show properly (1, 2, 0)

plt.show()


# Now we set up our network

class Network(nn.Module): # Remember to turn gradient graphing on before training!!!!!!!!!!!
    def __init__(self):
        super().__init__()
        # In_channel is the amount of color channels of our input image
        # kernal size is the size of the filter
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # pytorch layer object
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5) # pytorch layer object
        # the tensor going from the convolutional layers to the linear layers needs to be flattened
        # to calculate the output size of a convolution use:
        # ((InputSize + 2*pad - filtersize)/stride) + 1 # THis can be used to calculate the height and width separately
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)# pytorch layer object
        self.fc2 = nn.Linear(in_features=120, out_features=60)# pytorch layer object
        # out_features in the output layer needs to be = to the amount of classes in the dataset. MNIST fashion has 10
        # categories
        self.out = nn.Linear(in_features=60, out_features=10) # output layer # pytorch layer object

    def forward(self, t): # Never call the forward method directly. Using the network as a callable function is how its done
        # implement the forward pass
        # forward method will use all of the layers declared in our network
        # the forward method essentially maps the input tensor to the prediction tensors at the end of the network

        # (1) Input layer. This layer is usually implicit as it is an identity operation. Im including it for learning and clarity
        t = t

        # (2) Convolutional layer 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) Convolutional layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # we must now flatten our tensor before passing it to the first linear layer
        # remember that linear layers can only take flattened tensors as inputs
        # this flatten operation will be done in the 4th layer ebfore it is passed to the linear transformation

        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4) # With pytorch, passing a -1 tells the reshape function that it should determine what
                                  # the value should be based on the other values and the nuber of elements in our tensor
                                  # The 12 comes from the previous convolution layers output channels
                                  # The 4x4 is the original 28x28 image after it has gone through convolution and maxpooling
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer 2
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1) # Usually, a softmax function is used for the output layer but for this example, the
                                 # loss function that is being used is the cross entropy function, which implicitly performs
                                 # the softmax at the input layer

        return t

# the learnable parameters (or weights) are stored in the layers

network = Network()

print(network) # this only prints a string representation of the network because were inheriting nn.Module
                # This returns the __repr__ of the objects as defined by nn.module

print(network.conv1.weight) # prints the weights in that layer. We can do this for any of the layers present
                            # every weight tensor is an extension of the Pytorch Parameter(torch.Tensor) class

print(network.conv1.weight.shape) # in a convolutional layer, the weight is actually the filter values
                                  # outputs will have this matrix [6, 1, 5, 5] A batch of 6 tensors with 1 color chan dimension with a size of 5x5. So 6x1x5x5 tunable weights.
                                  # The weights match the kernal size defined when declaring the nn.conv2d object
print(network.conv2.weight.shape) # this will output [12,6,5,5] because we have 6 input channels and 12 filters being applied to each with each filter being 5x5 in size
print(network.fc1.weight.shape) # for linear layers, we have a 2x1 weight matrix. first index is output, second is input
print(network.fc2.weight.shape)

# note: .matmul() is used to multiply 2 matrices

# if we want to see the names and the values of the weights and biases in the layers
for name, param in network.named_parameters():
    print(name, '\t\t', param.shape)

# When we initialize a linear layer
print("fcx weights")
fcx = nn.Linear(in_features=4, out_features=3)#, bias=False) # We can also set the layer to have a bias or not. Default is true
# pytorch automatically creates a weight matrix based on the in and out parameters. That matrix is stored in fcx.
# This matrix is multiplies with the input matrix to form the output matrix. It is standard matrix multiplication
# weights x inputs = outputs. The weights in this case would be a 3x4 matrix
# to find out what the weights are in this layer
print(fcx.weight)

# We can manually set the weights of a layer

weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype=torch.float32)

fcx.weight = nn.Parameter(weight_matrix) # layer weights always need to be instances of Parameter class

print(fcx.weight)

# we can pass input features to a layer by doing the following

input_features = torch.tensor([
    [1,2,3,4]
], dtype=torch.float32)

print(fcx(input_features)) # Pytorch layer objects can be called like functions because they use a special method
                           # __call__() to define their behaviour when called. IMPORTANT: the call method invokes the
                           # forward() method for us.

# we created a network object earlier

# lets grab an image from the dataset

sample = next(iter(train_set))
image, label = sample
print("The image label is: " + str(label))
image.unsqueeze(0) # the image is currently [1,28,28]. 1 Color channel, 28 height and width
                   # pytorch neural nets need an input with a batch as well, so we need to unsqueeze the image at index 0
                   # to pop out an empty dimension to make it 1. The image is now [1,1,28,28]

# THis prediction is before training the network with the training set
print("Pre-training prediction.")
prediction = network(image.unsqueeze(0))
print(prediction) # returns a prediction tensor with the prediction values for all of the types of clothing
                  # It is a 1x9 tensor with more rows being added as more images come in

# If we want to see the class with the highest probability

print("Argmax output: " + str(prediction.argmax(dim=1))) # Dim essentially tells argmax to look at the row. dim 0 is column dim 1 is row

# if we want an actual percentage value

print("The softmax is: " + str(F.softmax(prediction, dim=1)))

# Before, we defined the data loader with a batch of 10. We can withdraw a batch of 10 images from the data loader like so

batch = next(iter(train_loader))

images, labels = batch  # We used the plural form here since we know that our loader is passing us 10 images

# Lets get the predictions for these 10 images

pred_new = network(images)
print(pred_new) # will give 10 arrays of length 10. 10 images with 10 output class. we get a prediction for each class

# We can now use the argmax function to get the index of the highest prediction for each image. The index will correspond
# to the label that the network is guessing

print(pred_new.argmax(dim=1)) # dim = 0 is the index that contains the image arrays. dim=1 contains the prediction values.
print(labels)

# one cool trick to see how many of the images the network guessed correctly

print(pred_new.argmax(dim=1).eq(labels)) # eq() is the equals function. It returns a 1 if the argmax is equal to the label, 0 if not
print(pred_new.argmax(dim=1).eq(labels).sum()) # adds up all the 1s. sum() simply does what you think
# If we want the number of correct guesses as an int, we need to retrieve the entry in the 1D sum() tensor using .item()
numCorrect = pred_new.argmax(dim=1).eq(labels).sum().item()

# Training the Neural Network

torch.set_grad_enabled(True) # We turned this off at the beginning of the script


# lets make a function that tells us how many predictions we got correct

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# lets take out a batch of 100 for this example

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader))
images, labels = batch

# -----Calculating the loss

preds = network(images)
loss = F.cross_entropy(preds, labels) # This line calculates the loss of the network using the cross entropy function
print(loss.item())

# remember that our batch is not representative right now. It contains 100 images of a relatively random order.
# we have no idea how many images are in each bin so were not sure if its a decent training set. This is for example
# purposes only

# ------Calculating Gradients

print(network.conv1.weight.grad) # We currently have no gradients stored in the network, as they were never calculated
loss.backward() # Backpropagation function
print(network.conv1.weight.grad.shape) # This tensor is the same shape as the weight tensor. This is because when we pass our
                                # dataset to the network, PyTorch begins tracking everyhting behind the scenes. When the
                                # functions are applied to the images, pytorch keeps track of this and all of the weights
                                # for that particular set of images. This is why when we pass preds, which is an output
                                # of network, to loss, pytorch still remembers the weights in that dataset so we can access
                                # them from any of the children in the process.

# ------Updating the weights
# we use an optimizer for this. Common ones are Adam and SGD

optimizer = optim.Adam(network.parameters(), lr=0.01) # lr is the learning rate. This is a hyperparameter that we need to fiddle with
                                                      # in order to try getting results more relevant to us.
                                                      # What can be useful is making this a variable defined at the top of the
                                                      # file that can be modified at will.
print(loss.item()) # Lets just check our pre grad loss again

print(get_num_correct(preds, labels)) # This is our self defined function that tells us how many guesses we got right
                                      # in our current batch of 100 images
optimizer.step() # This steps our paramters in the direction of the negative gradient, getting closer to the local minima
                 # this modifies the weights in the system

# lets pass the same images and get a new loss and prediction

preds = network(images)
loss = F.cross_entropy(preds, labels)
print(loss.item()) # our loss decreased from about 2.31 to 2.29

get_num_correct(preds,labels) # we now get 19 correct instead of 20

# we pass the network parameters to the optimizer so that it can update the weights during the training process
# the learning rate defines the "step" size. If it is too large, we will jump around the minimum without reaching it

# ----------- FULL TRAINING LOOP with training multiple epochs ---- what a close to final version wold look like
# epoch is a full pass over the full data set
# you want to keep running epochs until the accuracy starts to plateau


# commented out as it takes some time to train. ctrl+/ to uncomment

# network = Network() # network declaration and optimizer need to be outside of training loop
# optimizer = optim.Adam(network.parameters(), lr=0.01) # Using the Adam optimizer with a learning rate of 0.01
# train_loader = torch.utils.data.DataLoader(
#     train_set
#     ,batch_size=100
#     ,shuffle=True # Shuffled will shuffle the data at every epoch, so once the dataloader has been completely iterated through
# )
#
# for epoch in range(10): # We are doing 10 epochs. Number of epochs is a hyperparameter
#
#     total_loss = 0
#     total_correct = 0
#
#     for batch in train_loader: # Get Batch
#         images, labels = batch
#
#         preds = network(images) # Pass Batch
#         loss = F.cross_entropy(preds, labels) # Calculate Loss
#
#         optimizer.zero_grad() # we need to reset the gradients back to 0 or PyTorch will sum them every time theyre calculated. This is useful in some more advanced networks
#         loss.backward() # Calculate Gradients with backprop
#         optimizer.step() # Update Weights
#
#         total_loss += loss.item()
#         total_correct += get_num_correct(preds, labels) # Using the predefined function we made before
#
#     print(
#         "epoch", epoch,
#         "total_correct:", total_correct,
#         "loss:", total_loss
#    )

# The number of epochs is decided by us. Keep going until it plateaus is the basic strategy.


# ------- Confusion Matrix
# confusion matrix will allow us ot see which categories the NN is confusing with one another
# has predicted label on the x axis and true label on the y axis, and it builds a heat map to help visualize where the issues are happening

# we need length of training set
print(len(train_set))
print(len(train_set.targets))

@torch.no_grad() # This is a decoration that makes the function have grad off. If this wasnt here, everything would require a gradient which would increase computation tme by a lot
def get_all_preds(model, loader): # data loader is passed in so we can generate batches
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images) # we run the images through our model to get back our predictions
        all_preds = torch.cat( # we concatenate the predictions to our all_preds tensor
            (all_preds, preds)
            ,dim=0  # along the first dimension, row wise concat
        )
    return all_preds  # Tensor being returned with all the predictions


with torch.no_grad():  #we dont need gradient tracking becauser were not training in this case. Pytorch will automatically keep track of the tensor gradient graph, which adds overhead and computation time
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

print(train_preds.shape)

preds_correct = get_num_correct(train_preds, train_set.targets)

print('Total Correct: ' + str(preds_correct))
print('Total guessed: ' + str(preds_correct/len(train_set)))

stacked = torch.stack( # Note: concatenating joins a sequence of tensors along an existing axis, and stacking joins a sequence of tensors along a new axis
    (
        train_set.targets, # The targets are the actual labels of the images being passed
        train_preds.argmax(dim=1) # the predictions are what the model has guessed for each of the images
    )
    , dim=1 # We are stacking them column wise
)

# the pre stacked tensors had the labels in the shape [9, 0, 0, 3, ... , 1]
# post stack, they look like [ [9, 9] , [0, 0] , [0, 0] , [3, 3] , ...]

cmt = torch.zeros(10, 10, dtype=torch.int32) # we want a 10 by 10 tensor because we have 10 labels and we want 10 labels along the diagonal

for p in stacked: # p is the pair of label and guess
    j, k = p.tolist() # unpacking each so that they can be added to the confusion matrix
    cmt[j, k] = cmt[j, k] + 1 # incrementing the particular coordinate in the confusion matrix by 1. Remember that the goal is to have all guesses along the diag, which means that the guess matches the label

# sci kit learn also has built in confusion matrix generation if this becomes too hard

# now we can plot our confusion matrix

# code to plot the confusion matrix using matplotlib

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

names = (
    'T-shirt/top'
    ,'Trouser'
    ,'Pullover'
    ,'Dress'
    ,'Coat'
    ,'Sandal'
    ,'Shirt'
    ,'Sneaker'
    ,'Bag'
    ,'Ankle boot'
)

plt.figure(figsize=(10,10))
plot_confusion_matrix(cmt, names) # might need to refigure this out and iron out the code, it doesnt seem to show the graph


# look into tensorboard to visua