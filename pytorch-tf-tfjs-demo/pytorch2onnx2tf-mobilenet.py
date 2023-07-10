import os
import torch
import torch

proxy = os.environ.get('http_proxy1')

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

### Load Pytorch model

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

print("Pytorch: ",type(model(input_batch)))
print("Pytorch: ",(model(input_batch)))

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

tfmodel.export_graph("onnx_tf_data")
