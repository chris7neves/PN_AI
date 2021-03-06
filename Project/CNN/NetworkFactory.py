#---------------------------------------------------------------------
# This file will accept a list with all the hyperparameters entered #
# by the user in main. It will then use this list of hyperparameters#
# to define as many networks as necessary, train them, then return  #
# either the individual network objects or just the accuracy and    #
# performance of each network.										#
# E.G. Learning rates/Epoch numbers for networks                    #
#---------------------------------------------------------------------

import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.nn as nn

#class NetworkFactory: #TODO: implement the factory creator class
#    def __init__(self, params, dloader):


class basicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=9, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=11, stride=2)
        #TODO: Add more conv layers to reduce dimensionality (due to pooling layers

        # TODO: Recalculate input image size when using type 2 heatmap gen. Maybe have 2 sets of potential inputs and have a var to identify between both of them
        self.H = 27
        self.W = 37
        self.post_conv = 6 * self.H * self.W

        self.fc1 = nn.Linear(in_features=int(self.post_conv), out_features=int(self.post_conv))

        self.lin2 = int(self.post_conv * 0.3)
        self.fc2 = nn.Linear(in_features=self.post_conv, out_features=self.lin2)

        self.lin3 = int(self.lin2 * 0.3)
        self.fc3 = nn.Linear(in_features=self.lin2, out_features=self.lin3)

        self.lin4 = int(self.lin3 * 0.3)
        self.fc4 = nn.Linear(in_features=self.lin3, out_features=self.lin4)

        self.lin5 = int(self.lin4 * 0.3)
        self.fc5 = nn.Linear(in_features=self.lin4, out_features=self.lin5)

        # self.lin6 = int(self.lin5 * 0.3)
        # self.fc5 = nn.Linear(in_features=self.lin5, out_features=self.lin6)

        self.out = nn.Linear(in_features=self.lin5, out_features=2) # Final layer

    def forward(self, t):
        # Conv1
        t = self.conv1(t.float())
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Conv2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #print(f"Post Conv shape: {t.shape}")
        # Flatten Tensor to be compatible with Linear Layer
        t = t.reshape(-1, self.post_conv)

        # fc1
        t = self.fc1(t)
        t = F.relu(t)
        #print("Linear Layer 1 complete.")
        # fc2
        t = self.fc2(t)
        t = F.relu(t)
        #print("Linear Layer 2 complete.")
        # fc3
        t = self.fc3(t)
        t = F.relu(t)
        #print("Linear Layer 3 complete.")
        # fc4
        t = self.fc4(t)
        t = F.relu(t)
        #print("Linear Layer 4 complete.")
        # fc5
        t = self.fc5(t)
        t = F.relu(t)
        #print("Linear Layer 5 complete.")
        # fc5
        # t = self.fc6(t)
        # t = F.relu(t)
        # #print("Linear Layer 5 complete.")
        # out
        t = self.out(t)
        #print("output layer reached.")
        # There is no final max pool layer since adam optimizer is being used

        return t