##################################################
# feature_calc.py
# Utility used to calculate the output sizes of the
# height and width of the image after going through
# various layers.
# Used to simplify life
##################################################

ans = 1
layer_list = []

class layer:
    def __init__(self, type, filter, stride, pad=0):
        self.type = type
        self.filter = filter
        self.pad = pad
        self.stride = stride

    def pass_through(self, in_H, in_W):
        new_H = ((in_H + 2*self.pad - self.filter)/self.stride) + 1
        new_W = ((in_W + 2 * self.pad - self.filter) / self.stride) + 1
        return new_H, new_W


while ans != 0:
    ans = int(input("1 for conv, 2 for max pool and 3 for linear, 0 to exit: "))
    if ans == 1:
        fil, stride, pad = input("Please enter the conv filter size, stride and pad separated by a space: ").split()
        layer_list.append(layer('conv', int(fil), int(stride), int(pad)))
    if ans == 2:
        fil, stride, pad = input("Please enter the max pool filter size, stride and pad separated by a space: ").split()
        layer_list.append(layer('max_pool', int(fil), int(stride), int(pad)))
    if ans == 3:
        fil, stride, pad = input("Please enter the linear filter size, stride and pad separated by a space: ").split()
        layer_list.append(layer('linear', int(fil), int(stride), int(pad)))

input_H = 480
input_W = 640

for i, lay in enumerate(layer_list):
    temp_H = input_H
    temp_W = input_W
    input_H, input_W = lay.pass_through(input_H, input_W)
    print(f"Layer {i}, type: {lay.type}, filter size:{lay.filter}, stride: {lay.stride}, pad: {lay.pad}")
    print(f"Input H: {temp_H}, Input W: {temp_W}, Output H: {input_H}, Output W: {input_W}")

print(f"The resulting image size will be {input_H} x {input_W} (HxW)")
