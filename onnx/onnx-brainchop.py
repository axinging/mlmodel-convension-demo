import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np
import torch
import numpy as np
import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import numpy as np

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np


def dumpAsJson(name, input, inputShape):
    data = {'data': input}
    # data['operator'] = op
    # json_data = json.dumps(data)
    # print(json_data)
    with open(name+'.jsonc', 'w') as f:
        json_data = json.dump(data, f)
        # print(json_data)


def buildAndRunBinaryGraph(op, DATA_TYPE, comment):
    # Create the model and check
    inputShape = [1, 1, 256, 256, 256]
    outputChannel = 5
    channel = inputShape[1]
    inputChannel = inputShape[1]
    input = (np.random.rand(*tuple(inputShape))*10).astype('f').round(1)
    # Save the model
    #dumpAsJson('onnx-branchchop-case',input.flatten().tolist(), inputShape)
    with open('onnx-branchchop-case256.jsonc') as f:
        inputData = json.load(f)
        #conv3dncdhw
        input = arr = np.array(inputData['data'], dtype='float32').reshape(inputShape)
        MODEL_NAME = './brainchop/model_30_channels.onnx'
        #onnx.save(m1, MODEL_NAME)
        ort_sess = ort.InferenceSession(MODEL_NAME)
        outputs = ort_sess.run(None, {'input.1': input})
        # Print Result
        #print(type(DATA_TYPE).__name__, outputs[0])
        print(outputs[0].shape)
        ol = outputs[0][0].flatten().tolist()
        print(str(ol[0]) + ", " + str(ol[1])+ ", " + str(ol[2])+ ", " + str(ol[10])+ ", " + str(ol[100])+ ", " + str(ol[1000])+ ", " + str(ol[10000])+ ", " + str(ol[100000])+ ", " + str(ol[300000]))
        print(str(ol[len(ol)- 1])+ ", " + str(ol[len(ol)- 2])+ ", " + str(ol[len(ol)- 10])+ ", " + str(ol[len(ol)- 100])+ ", " + str(ol[len(ol)- 1000])+ ", " + str(ol[len(ol)- 10000])+ ", " + str(ol[len(ol)- 100000])+ ", " + str(ol[len(ol)- 300000]))


op = 'Sub'
DATA_TYPE = tp.FLOAT16
buildAndRunBinaryGraph(op, DATA_TYPE, 'FLOAT16')


