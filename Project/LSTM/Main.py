#
# Neural Network with Torch
#
from LSTM import Neural_Net

# X = [Hours Studied, Hours Sleeping]
# Y = [Grade]
X = Neural_Net.torch.tensor(([5, 9], [8, 8], [3, 6]), dtype=Neural_Net.torch.float)  # 3 x 2 tensor
Y = Neural_Net.torch.tensor(([92], [100], [69]), dtype=Neural_Net.torch.float)  # 3 x 1 tensor

# we can get the sizes of the tensor by using:
print(f"Size of X {X.size()}")
print(f"Size of Y {Y.size()}")
xPredicted = Neural_Net.torch.tensor(([0, 3]), dtype=Neural_Net.torch.float)  # 1 x 2 tensor of what we want to predict

#
# SCALING THE DATA
#

# we need to scale the data. Max function returns a tensor and the corresponding indices
# We use the "_" to capture the indices
X_max, _ = Neural_Net.torch.max(X, 0)  # Returns a tensor and an indice
xPredicted_max, _ = Neural_Net.torch.max(xPredicted, 0)  # Returns a tensor and an indice

X = Neural_Net.torch.div(X, X_max)
xPredicted = Neural_Net.torch.div(xPredicted, xPredicted_max)  # divide two tensors
Y = Y / 100  # Max test score is 100

NN = Neural_Net.ANN(2, 3, 1)
for i in range(1000):  # trains the NN 1,000 times
    # mean sum squared loss
    print("#" + str(i) + " Loss: " + str(Neural_Net.torch.mean((Y - NN(X)) ** 2).detach().item()))
    NN.train(X, Y)

NN.saveWeights(NN)
NN.predict(xPredicted)
