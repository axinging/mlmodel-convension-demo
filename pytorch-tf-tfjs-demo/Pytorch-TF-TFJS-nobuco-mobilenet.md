


# Pytorch model to TFJS model - mobilenetv2 (code)


### Step 1: Pytorch to Tensorflow (Code)

Run nobuco-pt-tf-demo.py to convert pytorch model to tensorflow model.
```
import os

import torch
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

proxy = 'http://proxy-'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


print(type(model))

pytorch_module = model

PATH_MODEL_ALL='torch_saved_model_all.pt'
torch.save(pytorch_module, PATH_MODEL_ALL)

model = torch.load(PATH_MODEL_ALL)
model.eval()
# print("All Loaded Result: ",model(dummy_image))


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


keras_model = nobuco.pytorch_to_keras(
    model,
    args=[input_batch], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)

print(type(keras_model))

keras_model.save('./torch_keras')
```

### Step 2: Tensorflow to TensorflowJS (File)

tensorflowjs_converter --input_format=tf_saved_model ./torch_keras ./predict_tfjs

### Step 3: Run TensorflowJS exmaple

```
<html>
<head>
  <title>TensorFlow.js demo</title>
</head>

<body>
  <h2>TensorFlow.js demo</h2>
  <script src="loader.js"></script>
  <script>
    'use strict';
    const tensorflow_DataType_DT_INT32 = 3;

    async function runTFJS() {
      const model = await tf.loadGraphModel('./simple/predict_houses_tfjs/model.json');
      const input = tf.ones([1, 3, 3, 1], 'float32');
      const output = model.predict(input);
      console.log(JSON.stringify(await output.data()));
    }

    (async function () {
      let localBuild = ['core', 'webgl', 'webgpu', 'tfjs-converter'];
      await loadTFJS(localBuild);
      await tf.setBackend('wasm');
      await tf.ready();
      tf.env().set('WEBGPU_CPU_FORWARD', false);
      await runTFJS();
    })();
  </script>
</body>
</html>
```

# Pytorch model to TFJS model - stable difussion - failed (File)

### Step 1 : Save stable difussion pytorch model as pt file: 
```
from diffusers import StableDiffusionPipeline

import torch

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
pipe = pipe.to("cpu")

prompt = "a photo of an astronaut riding a horse on mars"
#image = pipe(prompt).images[0]

#image.save("astronaut_rides_horse.png")

PATH_MODEL_ALL='torch_saved_model_all.pt'
torch.save(pipe, PATH_MODEL_ALL)

model = torch.load(PATH_MODEL_ALL)
print(type(model))
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse2.png")
```
### Step 2: Convert pt file as keras file (Failed): 

As demostrated in sd-pytorch-save-load.py,  save the whole stable difussion as a pt then load, it works.
But convert it to keras failed:

```
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import torch
import torch.nn as nn
import numpy as np

dummy_image = torch.ones((1, 1, 3, 3))

print("Input: "+ str(dummy_image))

PATH_MODEL_ALL='torch_saved_model_all.pt'
model = torch.load(PATH_MODEL_ALL)

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_image], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)

print(type(keras_model))
keras_model.save('./torch_keras')
```
