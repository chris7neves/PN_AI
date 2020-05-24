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


Mass_Data = SampleDataset()
trainloader = torch.utils.data.DataLoader(Mass_Data, batch_size=1, shuffle=True)


model = nn.Sequential(nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 2), nn.LogSoftmax(dim=1))
optimizer = torch.optim.SGD(model.parameters(), 0.003)
criterion = nn.NLLLoss()

epochs = 20000


for e in range(epochs):
    running_loss = 0
    print(e)
    for input, expected_output in trainloader:
        # Training pass
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, torch.max(expected_output, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Training loss: {running_loss/len(trainloader)}")


print("testing the network")
print("==================== 1 =======================")
input_tensor = torch.tensor([[6, 8]], dtype=torch.float)
output = model(input_tensor)
print(output)

print("==================== 2 =======================")
input_tensor = torch.tensor([[3, 5]], dtype=torch.float)
output = model(input_tensor)
print(output)


print("==================== 3 =======================")
input_tensor = torch.tensor([[1, 0]], dtype=torch.float)
output = model(input_tensor)
print(output)