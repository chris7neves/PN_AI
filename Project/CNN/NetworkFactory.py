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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=7, out_channels=14, kernel_size=5)

        self.fc1 = nn.Linear(in_features=14*117*157, out_features=257166)
        self.fc2 = nn.Linear(in_features=257166, out_features=192874) # 25% reduction
        self.fc3 = nn.Linear(in_features=192874, out_features=92437) # 50% reduction
        self.fc4 = nn.Linear(in_features=92437, out_features=48218) # 50% reduction
        self.fc5 = nn.Linear(in_features=48218, out_features=12054) # 75% reduction
        self.out = nn.Linear(in_features=12054, out_features=2) # Final layer

    def forward(self, t):
        # Conv1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Conv2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Flatten Tensor to be compatible with Linear Layer
        t = t.reshape(-1, 14*117*157)

        # fc1
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # fc3
        t = self.fc3(t)
        t = F.relu(t)

        # fc4
        t = self.fc4(t)
        t = F.relu(t)

        # fc5
        t = self.fc5(t)
        t = F.relu(t)

        # out
        t = self.out(t)

        # There is no final max pool layer since adam optimizer is being used

        return t