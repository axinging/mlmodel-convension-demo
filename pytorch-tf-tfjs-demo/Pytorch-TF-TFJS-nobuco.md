


# Pytorch model to TFJS model - simple example (code)


### Step 1: Pytorch to Tensorflow (Code)

Run nobuco-pt-tf-demo.py to convert pytorch model to tensorflow model.
```
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
```

### Step 2: Tensorflow to TensorflowJS (File)

tensorflowjs_converter --input_format=tf_saved_model ./ ./predict_houses_tfjs

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
