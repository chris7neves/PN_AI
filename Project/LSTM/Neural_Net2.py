import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import filesystem
import numpy as np
from datetime import datetime
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
        self.Log = filesystem.Logging("Mass_NN.txt")

    def forward(self, input_tensor):
        output = self.model(input_tensor)
        if (output[0][0] > output[0][1]):
            self.Log.print_and_log(f"Will pass")
        else:
            self.Log.print_and_log(f"Will get rekt")
        return output

    def prepdata(self, dataset, input_batch_size):
        self.trainloader = torch.utils.data.DataLoader(dataset, batch_size=input_batch_size, shuffle=True)

    def train(self, epochs):
        self.start_time = time.time()  # Start Timer
        for e in range(epochs):
            running_loss = 0
            self.Log.print_and_log(f"Epoch: {e}")
            for input, expected_output in self.trainloader:
                # Training pass
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, torch.max(expected_output, 1)[1])
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.Log.print_and_log(f"Training loss: {running_loss / len(self.trainloader)}")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.Log.print_and_log(f"Training Complete! Training Time: {self.elapsed_time}")

    def save(self, path, name):
        # https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        savefile = filesystem.os.path.join("E:/dev/PN_AI/Project/LSTM/Saved", f"{name}_model.pth")
        torch.save({'model_state_dict':self.model.state_dict()}, savefile)
        self.Log.print_and_log(f"Saved Neural Net to: {savefile}")


    def load(self, path, name):
        modelcheckpoint = torch.load(filesystem.os.path.join("E:/dev/PN_AI/Project/LSTM/Saved", f"{name}_model.pth"))
        self.model.load_state_dict(modelcheckpoint['model_state_dict'])
        self.Log.print_and_log(f"Loaded Neural Net from {name}_model.pth")


    def test(self):
        print("testing network")
        for input, expected_output in self.validationloader:
            print("input")
            print(input)
            print("expected output")
            print(expected_output)
            output = self.model(input)
            print("Output from Network")
            print(output)
            break

            # value = output
            # print(value)
            # if expected_output[0] > expected_output[1]:
            #     print("Subject: TYP")
            #     if output[0][0] > output[0][1]:
            #         print("Guess: TYP - CORRECT :)")
            #     else:
            #         print("Guess: ASD - INCORRECT :(")
            # else:
            #     print("Subject: ASD")
            #     if output[0][0] < output[0][1]:
            #         print("Guess: ASD - CORRECT :)")
            #     else:
            #         print("Guess: TYP - INCORRECT :(")


    def splitandprepdata(self, dataset, validation_ratio, batch_size, shuffle):
        validation_split = validation_ratio
        dataset_size = dataset.InputDimensions[0]
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        print(f"Dataset size = {dataset_size}")
        print(f"validation split * dataset size = {validation_split * dataset_size}")
        print(f"floored split {split}")

        if shuffle:
           np.random.seed(datetime.now())
           np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        self.validationloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)





