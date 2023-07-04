import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x

data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0,8.0,9.0]]
dummy_image = torch.ones((1, 1, 3, 3))



# dummy_image = torch.rand(size=(1, 1, 3, 3))

print((MyModule()(dummy_image)))

pytorch_module = MyModule().eval()

print('Weight: ',pytorch_module.conv.weight)

print("Reslut : "+ str(dummy_image))
print(pytorch_module(dummy_image))


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
