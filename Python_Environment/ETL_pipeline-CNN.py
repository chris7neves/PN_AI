#---------------------------------------------------------------------
# This file will deal with the import of the data, cleaning        #
# and manipulation of the heat map images, resolution changes,      #
# and preparation of the images before being used by the network.   #
#---------------------------------------------------------------------

# Imports

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

width = 1
height = 1