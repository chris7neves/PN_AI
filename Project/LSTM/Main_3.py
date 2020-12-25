from LSTM import Neural_Net2
from Response_DataLoader import ResponseDataset
from filesystem import os

epochs = 10000
batch_size = 5
Mass_Data = ResponseDataset()

model = Neural_Net2.nn.Sequential(Neural_Net2.nn.Linear(24, 24), Neural_Net2.nn.Sigmoid(),
                                  Neural_Net2.nn.Linear(24, 15), Neural_Net2.nn.BatchNorm1d(15, 2),
                                  Neural_Net2.nn.Sigmoid(),
                                  Neural_Net2.nn.Linear(15, 2), Neural_Net2.nn.LogSoftmax(dim=1))

nn = Neural_Net2.ANN(model, 0.006)

#array = Mass_Data.GetScatterData()
#nn.EDA(array)
nn.splitandprepdata(Mass_Data, 0.2, batch_size, True)
#nn.prepdata(Mass_Data, batch_size)
nn.train(epochs)

nn.test()
#
# nn.save("E:/dev/PN_AI/Project/LSTM/Saved", "massimo")
