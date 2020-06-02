from LSTM import Neural_Net2
from Response_DataLoader import ResponseDataset
from filesystem import os

epochs = 100000
batch_size = 1
Mass_Data = ResponseDataset()

model = Neural_Net2.nn.Sequential(Neural_Net2.nn.Linear(176, 200), Neural_Net2.nn.Sigmoid(),
                                  Neural_Net2.nn.Linear(200, 75), Neural_Net2.nn.Sigmoid(),
                                  Neural_Net2.nn.Linear(75, 2), Neural_Net2.nn.LogSoftmax(dim=1))

nn = Neural_Net2.ANN(model, 0.003)
nn.prepdata(Mass_Data, batch_size)
nn.train(epochs)

nn.test()

nn.save("E:/dev/PN_AI/Project/LSTM/Saved", "massimo")



