
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


def dumpAsJson(name, op, input, inputShape, filter, filterShape, kernel_size, output, outputShape, stride, dilation, padding, useBias, bias, biasShape):
    data = {}
    biasStr = ''
    if bias is not None:
        biasStr = ' with bias'
    data['name'] = name + biasStr + ', x=' + ''.join(str(inputShape)) + ', f=' + ''.join(
        str(filterShape)) + ', s='+str(stride) + ', d='+str(dilation) + ', p='+str(padding)
    data['operator'] = op
    print(padding)
    if (padding.upper() == 'SAME'):
        padding = 'SAME_UPPER'
    data['attributes'] = [
        {"name": "kernel_shape", "data": kernel_size, "type": "ints"},
        {"name": "auto_pad", "data": padding.upper(), "type": "string"},
        {"name": "strides", "data": [1, 1, 1], "type": "ints"},
        {"name": "dilations", "data": [1, 1, 1], "type": "ints"}
    ]
    if useBias == 1:
        data['cases'] = [
            {
                "name": "T[0]",
                "inputs": [
                    {
                        "data": input,
                        "dims": inputShape,
                        "type": "float32"
                    },
                    {
                        "data": filter,
                        "dims": filterShape,
                        "type": "float32"
                    },
                    {
                        "data": bias,
                        "dims": biasShape,
                        "type": "float32"
                    }
                ],
                "outputs": [
                    {
                        "data": output,
                        "dims": outputShape,
                        "type": "float32"
                    }
                ]
            }
        ]
    else:
        data['cases'] = [
            {
                "name": "T[0]",
                "inputs": [
                    {
                        "data": input,
                        "dims": inputShape,
                        "type": "float32"
                    },
                    {
                        "data": filter,
                        "dims": filterShape,
                        "type": "float32"
                    }
                ],
                "outputs": [
                    {
                        "data": output,
                        "dims": outputShape,
                        "type": "float32"
                    }
                ]
            }
        ]
    # data['operator'] = op
    # json_data = json.dumps(data)
    # print(json_data)
    with open(name+'.jsonc', 'w') as f:
        json_data = json.dump([data], f)
        # print(json_data)


inputShape = [1, 1, 4, 4, 4]
outputChannel = 5
channel = inputShape[1]
inputChannel = inputShape[1]
input = (np.random.rand(*tuple(inputShape))*10).astype('f').round(1)
# print(input.flatten())
# out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
# kernel_size = (inputShape[2], inputShape[3], inputShape[4])
kernel_size = (3, 3, 3)
filterShape = [outputChannel, channel,
               kernel_size[0], kernel_size[1], kernel_size[2]]
outChannel = filterShape[0]
filter = (np.random.rand(*tuple(filterShape))*10).astype('f').round(1)
# print(filter.flatten())
useBias = 1
biasShape = [outChannel]
# bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)
bias = (np.random.rand(*tuple(biasShape))*10).astype('f').round(1)
# kernel_size = (filterShape[2], filterShape[3],filterShape[4])
stride = 1
dilation = 1
padding = 'same'
paddingOX = padding
if (padding.upper() == 'SAME'):
    paddingOX = 'SAME_UPPER'


@script()
def Conv(X, w, bias):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    # print(padding)
    return op.Conv(X, w, bias, auto_pad=paddingOX, dilations=[dilation, dilation, dilation], group=1, kernel_shape=kernel_size)


def ConvOne(v, w, bias, padding):
    if (padding.upper() == 'SAME'):
        padding = 'SAME_UPPER'
    result1 = Conv(v, w, bias, padding)
    # print(*result1.flatten(),sep=', ')
    return result1

# np.random.rand(3,2)


def runPytorch():
    # NCDHW
    m = nn.Conv3d(inputChannel, outChannel, kernel_size,
                  stride=stride, dilation=dilation, padding=padding)
    m.bias = torch.nn.Parameter(
        torch.from_numpy(bias).float().reshape(biasShape))
    m.weight = nn.Parameter(torch.from_numpy(
        filter).float().reshape(filterShape))
    # NCDHW
    # m = m.to(memory_format=torch.channels_last_3d)
    # non-square kernels and unequal stride and with padding
    inputT = torch.from_numpy(input).reshape(inputShape)
    print(inputT.shape)
    inputT = inputT.to(memory_format=torch.channels_last_3d)
    print(inputT.shape)
    with torch.no_grad():
        output = m(inputT)
    print("Conv in ONNX:")
    print(type(output))  # outShape = [1, 2, 1, 1, 1]
    # print((output.flatten()))

    dumpAsJson('conv3dpt', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size,
               output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), useBias, bias.flatten().tolist(), biasShape)
    # dumpAsJson('conv3d', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size, output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), None, biasShape)
    return torch.Tensor.numpy(output)


def runOnnsScript():
    output = ConvOne(input, filter, bias, padding)
    print("Conv in ONNX:")
    print(type(output))
    dumpAsJson('conv3dox', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size,
               output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), useBias, bias.flatten().tolist(), biasShape)
    return output


outOS = runOnnsScript()

outPT = runPytorch()

print(type(outOS))
print(type(outPT))
print(np.allclose(outOS, outPT))
# .
print(np.array_equiv(outOS, outPT))
