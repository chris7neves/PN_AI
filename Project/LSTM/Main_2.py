from LSTM import Neural_Net2
from Custom_DataLoader import SampleDataset

epochs = 2000
batch_size = 1
Mass_Data = SampleDataset()

nn = Neural_Net2.ANN(2, 3, 2, 0.003)
nn.prepdata(Mass_Data, batch_size)
nn.train(epochs)

input_tensor = Neural_Net2.torch.tensor([[8, 8]], dtype=Neural_Net2.torch.float)
output = nn.forward(input_tensor)
print(output)


