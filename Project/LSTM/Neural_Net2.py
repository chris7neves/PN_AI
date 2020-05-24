import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

class ANN(nn.Module):
    def __init__(self, net_model, learning_rate_input):
        super(ANN, self).__init__()
        # parameters
        self.learning_rate = learning_rate_input
        self.criterion = nn.NLLLoss()
        self.model = nn.Sequential(nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 2), nn.LogSoftmax(dim=1))
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
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