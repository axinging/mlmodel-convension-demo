from transformers import AutoTokenizer, CLIPTextModel
import os

import torch
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

proxy = 'http://proxy.proxy.com:91'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


print(type(model))

pytorch_module = model

PATH_MODEL_ALL='mobilenet_v2.pt'
torch.save(pytorch_module, PATH_MODEL_ALL)

model = torch.load(PATH_MODEL_ALL)
model.eval()


###  Pytorch2ONNX

import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

print("Pytorxh: ",type(model(input_batch)))
print("Pytorxh: ",(model(input_batch)))

ONNX_FILE = 'mobilenet_v2.onnx'
# Export the model
torch.onnx.export(model,               # model being run
                  input_batch,                         # model input (or a tuple for multiple inputs)
                  ONNX_FILE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

### ONNX2TF

import onnx

onnx_model = onnx.load(ONNX_FILE)
onnx.checker.check_model(onnx_model)
from onnx_tf.backend import prepare

tfmodel = prepare(onnx_model)  # run the loaded model
output = tfmodel.run(input_batch)  # run the loaded model

print("TFModel: ", type(tfmodel))
# print("TF: ", output)
print("TF: ", type(output))

tfmodel.export_graph("oonx_tf_data")

wp-40/workspace/project/xing/mlmodel-convension-demo/cliptextmodel
wp >>> cat pytorch2tfjs-mobilenet.py                                                                                                                                                                23-07-06 13:32
from transformers import AutoTokenizer, CLIPTextModel
import os

import torch
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer



import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


print(type(model))

pytorch_module = model

PATH_MODEL_ALL='mobilenet_v2.pt'
torch.save(pytorch_module, PATH_MODEL_ALL)

model = torch.load(PATH_MODEL_ALL)
model.eval()


###  Pytorch2ONNX

import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

print("Pytorxh: ",type(model(input_batch)))
print("Pytorxh: ",(model(input_batch)))

ONNX_FILE = 'mobilenet_v2.onnx'
# Export the model
torch.onnx.export(model,               # model being run
                  input_batch,                         # model input (or a tuple for multiple inputs)
                  ONNX_FILE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

### ONNX2TF

import onnx

onnx_model = onnx.load(ONNX_FILE)
onnx.checker.check_model(onnx_model)
from onnx_tf.backend import prepare

tfmodel = prepare(onnx_model)  # run the loaded model
output = tfmodel.run(input_batch)  # run the loaded model

print("TFModel: ", type(tfmodel))
# print("TF: ", output)
print("TF: ", type(output))

tfmodel.export_graph("oonx_tf_data")
