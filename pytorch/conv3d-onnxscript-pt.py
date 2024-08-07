
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


def dumpCon3dAsJson(name, suffix, op, input, inputShape, filter, filterShape, kernel_size, output, outputShape, stride, dilation, padding, paddingNum, useBias, bias, biasShape, dtype):
    data = {}
    biasStr = ''
    if dtype == 'f':
        dtype = 'float32'
    elif dtype == 'f2':
        dtype = 'float16'
    if useBias == 1:
        biasStr = ' with bias'
    if (padding.upper() == 'SAME'):
        padding = 'SAME_UPPER'
    print(type(paddingNum))
    if (type(paddingNum) == 'tuple'):
        paddingNum = list(paddingNum)
    print(type(paddingNum))
    data['name'] = name + biasStr + ', x=' + ''.join(str(inputShape)) + ', f=' + ''.join(
        str(filterShape)) + ', s='+str(stride) + ', d='+str(dilation) + ', auto_pad='+str(padding) + ', pads='+str(paddingNum)
    data['operator'] = op
    data['attributes'] = [
        {"name": "kernel_shape", "data": kernel_size, "type": "ints"},
        {"name": "auto_pad", "data": padding.upper(), "type": "string"},
        {"name": "strides", "data": [1, 1, 1], "type": "ints"},
        {"name": "pads", "data": paddingNum, "type": "ints"},
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
                        "type": dtype
                    },
                    {
                        "data": filter,
                        "dims": filterShape,
                        "type": dtype
                    },
                    {
                        "data": bias,
                        "dims": biasShape,
                        "type": dtype
                    }
                ],
                "outputs": [
                    {
                        "data": output,
                        "dims": outputShape,
                        "type": dtype
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
                        "type": dtype
                    },
                    {
                        "data": filter,
                        "dims": filterShape,
                        "type": dtype
                    }
                ],
                "outputs": [
                    {
                        "data": output,
                        "dims": outputShape,
                        "type": dtype
                    }
                ]
            }
        ]
    # data['operator'] = op
    # json_data = json.dumps(data)
    # print(json_data)
    with open(name+suffix+'.jsonc', 'w') as f:
        json_data = json.dump([data], f)
        # print(json_data)


# Place holder data.
inputShape = [1, 2, 4, 4, 4]
outputChannel = 1
channel = inputShape[1]
inputChannel = inputShape[1]
dtype = 'f2'
input = (np.random.rand(*tuple(inputShape))*10).astype(dtype).round(1)
print(input)
# print(input.flatten())
# out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
# kernel_size = (inputShape[2], inputShape[3], inputShape[4])
kernel_size = (1, 1, 1)
filterShape = [outputChannel, channel,
               kernel_size[0], kernel_size[1], kernel_size[2]]
# outChannel = filterShape[0]
filter = (np.random.rand(*tuple(filterShape))*10).astype(dtype).round(1)
# print(filter.flatten())
useBias = 1
biasShape = [outputChannel]
# bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)
# np.float16
bias = (np.random.rand(*tuple(biasShape))*10).astype(dtype).round(1)
# kernel_size = (filterShape[2], filterShape[3],filterShape[4])
stride = 1
dilation = 1
padding = 'NOTSET'
paddingOX = padding
paddingPT = padding
paddingNumOX = [2, 2, 2, 2, 2, 2]
paddingNumPT = (2, 2, 2)
# paddingNumOX = [1, 1, 1, 1, 1, 1]
# paddingNumPT = (1, 1, 1)

if (padding.upper() == 'SAME'):
    paddingOX = 'SAME_UPPER'


@script()
def Conv(X, w, bias):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    # print(padding)
    return op.Conv(X, w, bias, auto_pad=paddingOX, dilations=[dilation, dilation, dilation], group=1, kernel_shape=kernel_size)


@script()
def ConvPadNumber(X, w, bias):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    # print(padding)
    return op.Conv(X, w, bias, auto_pad='NOTSET', dilations=[dilation, dilation, dilation], group=1, kernel_shape=kernel_size, pads=paddingNumOX)


def ConvOne(v, w, bias):
    result1 = Conv(v, w, bias) if (
        padding != 'NOTSET') else ConvPadNumber(v, w, bias)
    # print(*result1.flatten(),sep=', ')
    return result1


def runPytorch():
    # NCDHW
    # 20 if someBoolValue else num1
    m = nn.Conv3d(inputChannel, outputChannel, kernel_size, stride=stride, dilation=dilation, padding=paddingPT) if (
        padding != 'NOTSET') else nn.Conv3d(inputChannel, outputChannel, kernel_size, stride=stride, dilation=dilation, padding=paddingNumPT)
    m.bias = torch.nn.Parameter(
        torch.from_numpy(bias).reshape(biasShape))
    if dtype == 'f2':
        m.bias.to(torch.float16)
    m.weight = nn.Parameter(torch.from_numpy(
        filter).reshape(filterShape))
    if dtype == 'f2':
        m.weight.to(torch.float16)
    # NCDHW
    # m = m.to(memory_format=torch.channels_last_3d)
    # non-square kernels and unequal stride and with padding
    inputT = torch.from_numpy(input).reshape(inputShape)
    print(inputT.shape)
    inputT = inputT.to(memory_format=torch.channels_last_3d)
    print(inputT.shape)
    with torch.no_grad():
        output = m(inputT)
    # print("Conv in PT:")
    # print(type(output))  # outShape = [1, 2, 1, 1, 1]
    # print((output.flatten()))

    dumpCon3dAsJson('conv3d', 'pt', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size,
                    output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), paddingNumPT, useBias, bias.flatten().tolist(), biasShape, dtype)
    # dumpCon3dAsJson('conv3d', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size, output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), None, biasShape)
    return torch.Tensor.numpy(output)


def runOnnxScript():
    output = ConvOne(input, filter, bias)
    print("Conv in ONNX nnn:")
    print(type(output))
    print(output.shape)
    dumpCon3dAsJson('conv3d', 'ox', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size,
                    output.flatten().tolist(), output.shape, stride, dilation, paddingOX.upper(), paddingNumOX, useBias, bias.flatten().tolist(), biasShape, dtype)
    return output


outOS = runOnnxScript()

outPT = runPytorch()

print(type(outOS))
print(type(outPT))
print(np.allclose(outOS, outPT))
# .
print(np.array_equiv(outOS, outPT))
