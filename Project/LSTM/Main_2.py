from LSTM import Neural_Net2
from Custom_DataLoader import SampleDataset

epochs = 2000
batch_size = 1
Mass_Data = SampleDataset()

model = Neural_Net2.nn.Sequential(Neural_Net2.nn.Linear(2, 3), Neural_Net2.nn.Sigmoid(), Neural_Net2.nn.Linear(3, 2), Neural_Net2.nn.LogSoftmax(dim=1))
nn = Neural_Net2.ANN(model, 0.003)
nn.prepdata(Mass_Data, batch_size)
nn.train(epochs)

input_tensor = Neural_Net2.torch.tensor([[3, 8]], dtype=Neural_Net2.torch.float)
output = nn.forward(input_tensor)
print(output)



