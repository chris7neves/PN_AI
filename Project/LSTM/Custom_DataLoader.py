import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SampleDataset(Dataset):  # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    def __init__(self):
        self.X = torch.tensor(([5, 9], [8, 8], [3, 6], [2, 8], [4, 6], [4, 5.5], [5.5, 8], [6, 4]), requires_grad=True,
                               dtype=torch.float)  # 3 x 2 tensor
        # self.Y = torch.tensor(([92], [100], [69], [90], [79], [84], [86], [75]), dtype=torch.float)  # 3 x 1 tensor
        self.Y = torch.tensor(([1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]), dtype=torch.float)  # [Pass, Fail]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x_data = self.X[index]
        y_data = self.Y[index]
        return x_data, y_data