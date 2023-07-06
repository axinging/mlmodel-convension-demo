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

input_names = [ "actual_input" ]
output_names = [ "output" ]

dummy_input = torch.randn(1, 1, 3, 3)
PATH = 'torch_saved_model.pt'
torch.save(pytorch_module.state_dict(), PATH)

model = MyModule().eval()
model.load_state_dict(torch.load(PATH))
model.eval()
print("Loaded Result: ",model(dummy_image))


PATH_MODEL_ALL='torch_saved_model_all.pt'
torch.save(pytorch_module, PATH_MODEL_ALL)

model = torch.load(PATH_MODEL_ALL)
model.eval()
print("All Loaded Result: ",model(dummy_image))


keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_image], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)

print(type(keras_model))

keras_model.save('./torch_keras_frompt')

