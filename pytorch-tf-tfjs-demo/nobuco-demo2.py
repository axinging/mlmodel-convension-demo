import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.relu((x))
        return x

data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0,8.0,9.0]]
dummy_image = torch.tensor([[[[-1., -1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]]])

print("Result without eval: ",(MyModule()(dummy_image)))
print('\n\nWeight and Bias parameters:')

pytorch_module = MyModule().eval()
for param in pytorch_module.parameters():
    print(param)


print(list(pytorch_module.parameters()))

print("Input : ", str(dummy_image))
print("Result : ", pytorch_module(dummy_image))


keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[dummy_image], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)

print(type(keras_model))

keras_model.save('./torch_keras')

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)
print(type(lin))

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
