import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import torch
import torch.nn as nn
import numpy as np

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x

dummy_image = torch.ones((1, 1, 3, 3))

pytorch_module = MyModule().eval()

print('Parameters: ', list(pytorch_module.parameters()))
print("Input: "+ str(dummy_image))
print("Result: ",pytorch_module(dummy_image))


keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[dummy_image], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)

print(type(keras_model))

keras_model.save('./torch_keras')
