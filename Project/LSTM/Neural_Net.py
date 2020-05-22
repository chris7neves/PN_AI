import torch
import torch.nn as nn

# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
# https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

class ANN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(ANN, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        # weights
        # self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 2 X 3 tensor
        # self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 3 X 1 tensor
        self.W1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.W2 = nn.Linear(self.hiddenSize, self.outputSize)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        o = self.W1(X)
        o = self.sigmoid(o)
        o = self.W2(o)
        o = self.softmax(o)
        return o

        #self.z = torch.matmul(X, self.W1)  # 3 X 3 ".dot" does not broadcast in PyTorch
        #self.z2 = self.sigmoid(self.z)  # activation function
        #self.z3 = torch.matmul(self.z2, self.W2)
        #o = self.sigmoid(self.z3)  # final activation function
        #return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self,tensor):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(tensor))
        print("Output: \n" + str(self.forward(tensor)))

