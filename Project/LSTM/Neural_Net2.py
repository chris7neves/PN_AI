import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import filesystem
from collections import OrderedDict

# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

class ANN(nn.Module):
    def __init__(self, net_model, learning_rate_input):
        super(ANN, self).__init__()
        # parameters
        self.learning_rate = learning_rate_input
        self.criterion = nn.NLLLoss()
        #self.model = nn.Sequential(nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 2), nn.LogSoftmax(dim=1))
        self.model = net_model
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        #SGD vs crossentropy
        # Timers
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

    def forward(self, input_tensor):
        output = self.model(input_tensor)
        if (output[0][0] > output[0][1]):
            print("Will pass")
        else:
            print("Will get rekt")
        return output

    def prepdata(self, dataset, input_batch_size):
        self.trainloader = torch.utils.data.DataLoader(dataset, batch_size=input_batch_size, shuffle=True)

    def train(self, epochs):
        self.start_time = time.time()  # Start Timer
        for e in range(epochs):
            running_loss = 0
            print(e)
            for input, expected_output in self.trainloader:
                # Training pass
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, torch.max(expected_output, 1)[1])
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Training loss: {running_loss / len(self.trainloader)}")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Training Complete! Training Time: {self.elapsed_time}")

    def save(self, path, name):
        # https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'model_state_dict':self.model.state_dict(),
            }, filesystem.os.path.join("E:/dev/PN_AI/Project/LSTM/Saved", f"{name}_model.pth"))
        # torch.save(self.model, filesystem.os.path.join("E:/dev/PN_AI/Project/LSTM/Saved", f"{name}_model.pth"))


    def load(self, path, name):
        modelcheckpoint = torch.load(filesystem.os.path.join("E:/dev/PN_AI/Project/LSTM/Saved", f"{name}_model.pth"))
        self.model.load_state_dict(modelcheckpoint['model_state_dict'])
        #self.model.load_state_dict(modelcheckpoint['optimizer_state_dict'])




