from LSTM import Neural_Net2
from Custom_DataLoader import SampleDataset
from filesystem import os

epochs = 200
batch_size = 1
Mass_Data = SampleDataset()

model = Neural_Net2.nn.Sequential(Neural_Net2.nn.Linear(2, 3), Neural_Net2.nn.Sigmoid(), Neural_Net2.nn.Linear(3, 2), Neural_Net2.nn.LogSoftmax(dim=1))

nn = Neural_Net2.ANN(model, 0.003)
nn.prepdata(Mass_Data, batch_size)
nn.train(epochs)

input_tensor = Neural_Net2.torch.tensor([[3, 8]], dtype=Neural_Net2.torch.float)
output = nn.forward(input_tensor)
nn.Log.print_and_log("Neural Net 1")
print(output)

nn.save("E:/dev/PN_AI/Project/LSTM/Saved", "massimo")

# ================ Loading into a second NN ========================

model2 = Neural_Net2.nn.Sequential(Neural_Net2.nn.Linear(2, 3), Neural_Net2.nn.Sigmoid(), Neural_Net2.nn.Linear(3, 2), Neural_Net2.nn.LogSoftmax(dim=1))
nn2 = Neural_Net2.ANN(model2, 0.003)

nn2.load("E:/dev/PN_AI/Project/LSTM/Saved", "massimo")
output2 = nn2.forward(input_tensor)
nn2.Log.print_and_log("Neural Net 2")
print(output)


# import torch
#
# X = torch.tensor(([5, 9], [8, 8], [3, 6], [2, 8], [4, 6], [4, 5.5], [5.5, 8], [6, 4]), requires_grad=True,
#                       dtype=torch.float)  # 3 x 2 tensor
# # self.Y = torch.tensor(([92], [100], [69], [90], [79], [84], [86], [75]), dtype=torch.float)  # 3 x 1 tensor
# Y = torch.tensor(([1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]), dtype=torch.float)  # [Pass, Fail]
#
# print(X)
#
#
# #X = X / 10
# print(X[0])

