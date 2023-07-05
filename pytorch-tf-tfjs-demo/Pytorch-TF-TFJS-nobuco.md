


# Pytorch model to TFJS model (code)


### Step 1: Pytorch to Tensorflow

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

### Step 2: Tensorflow to TensorflowJS

tensorflowjs_converter --input_format=tf_saved_model ./ ./predict_houses_tfjs

### Run TensorflowJS exmaple

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
