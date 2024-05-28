
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


def dumpAsJson(name, suffix, op, input, inputShape, filter, filterShape, kernel_size, output, outputShape, stride, dilation, padding, useBias, bias, biasShape):
    data = {}
    biasStr = ''
    if useBias == 1:
        biasStr = ' with bias'
    if (padding.upper() == 'SAME'):
        padding = 'SAME_UPPER'
    data['name'] = name + biasStr + ', x=' + ''.join(str(inputShape)) + ', f=' + ''.join(
        str(filterShape)) + ', s='+str(stride) + ', d='+str(dilation) + ', p='+str(padding)
    data['operator'] = op
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
    with open(name+suffix+'.jsonc', 'w') as f:
        json_data = json.dump([data], f)
        # print(json_data)

# Place holder data.
inputShape = [1, 3, 4, 4, 4]
outputChannel = 5
channel = inputShape[1]
inputChannel = inputShape[1]
input = (np.random.rand(*tuple(inputShape))*10).astype('f').round(1)
# print(input.flatten())
# out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
# kernel_size = (inputShape[2], inputShape[3], inputShape[4])
kernel_size = (1, 1, 1)
filterShape = [outputChannel, channel,
               kernel_size[0], kernel_size[1], kernel_size[2]]
#outChannel = filterShape[0]
filter = (np.random.rand(*tuple(filterShape))*10).astype('f').round(1)
# print(filter.flatten())
useBias = 1
biasShape = [outputChannel]
# bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)
bias = (np.random.rand(*tuple(biasShape))*10).astype('f').round(1)
# kernel_size = (filterShape[2], filterShape[3],filterShape[4])
stride = 1
dilation = 1
padding = 'same'
paddingOX = padding
paddingPT = padding
if (padding.upper() == 'SAME'):
    paddingOX = 'SAME_UPPER'

outputWeb = input

@script()
def Conv(X, w, bias):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    # print(padding)
    return op.Conv(X, w, bias, auto_pad=paddingOX, dilations=[dilation, dilation, dilation], group=1, kernel_shape=kernel_size)


def ConvOne(v, w, bias):
    result1 = Conv(v, w, bias)
    # print(*result1.flatten(),sep=', ')
    return result1


def runPytorch():
    # NCDHW
    m = nn.Conv3d(inputChannel, outputChannel, kernel_size,
                  stride=stride, dilation=dilation, padding=paddingPT)
    m.bias = torch.nn.Parameter(
        torch.from_numpy(bias).float().reshape(biasShape))
    m.weight = nn.Parameter(torch.from_numpy(
        filter).float().reshape(filterShape))
    # NCDHW
    # m = m.to(memory_format=torch.channels_last_3d)
    # non-square kernels and unequal stride and with padding
    inputT = torch.from_numpy(input).reshape(inputShape)
    #print(inputT.shape)
    inputT = inputT.to(memory_format=torch.channels_last_3d)
    #print(inputT.shape)
    with torch.no_grad():
        output = m(inputT)
    #print("Conv in PT:")
    #print(type(output))  # outShape = [1, 2, 1, 1, 1]
    # print((output.flatten()))

    dumpAsJson('conv3d', 'pt', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size,
               output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), useBias, bias.flatten().tolist(), biasShape)
    # dumpAsJson('conv3d', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size, output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), None, biasShape)
    return torch.Tensor.numpy(output)


def runOnnxScript():
    output = ConvOne(input, filter, bias)
    #print("Conv in ONNX nnn:")
    #print(type(output))
    #print(output.shape)
    dumpAsJson('conv3d', 'ox', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size,
               output.flatten().tolist(), output.shape, stride, dilation, paddingOX.upper(), useBias, bias.flatten().tolist(), biasShape)
    return output



'''
      { "name": "kernel_shape", "data": [2, 1, 2], "type": "ints" },
      { "name": "auto_pad", "data": "VALID", "type": "string" },
      { "name": "strides", "data": [1, 1, 1], "type": "ints" },
      { "name": "dilations", "data": [1, 1, 1], "type": "ints" }

          "cases": [
      {
        "name": "T[0]",
        "inputs": [
          {
            "data": [0.25, 0.5, 0.75, 1],
            "dims": [1, 1, 2, 1, 2],
            "type": "float32"
          },
          {
            "data": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
            "dims": [2, 1, 2, 1, 2],
            "type": "float32"
          }
        ],
        "outputs": [
          {
            "data": [0.9375, 2.1875],
            "dims": [1, 2, 1, 1, 1],
            "type": "float32"
          }
        ]
      }
    ]
'''



with open('conv3dncdhw.jsonc') as f:
    #caseIndex = 4
    jsoncData = json.load(f)
    for caseIndex in range(len(jsoncData)):
        #conv3dncdhw
        if (caseIndex !=9):
            continue
        print('case ' + str(caseIndex) + ", " + str(jsoncData[caseIndex]['name']))
        attributes = jsoncData[caseIndex]['attributes']
        print(len(jsoncData))
        cases = jsoncData[caseIndex]['cases']
        kernel_shape = attributes[0]['data']
        kernel_size = kernel_shape
        auto_pad = attributes[1]['data']

        padding = auto_pad.lower()
        paddingPT = auto_pad.lower()
        paddingOX = auto_pad.upper()
        if (paddingOX == 'SAME'):
            paddingOX = 'SAME_UPPER'
    
        if (padding.upper() == 'SAME_UPPER'):
            paddingPT = 'same'


        stride = attributes[2]['data'][0]
        dilations = attributes[3]['data'][0]
        #print('kernel_shape =' + str(kernel_shape) + ',auto_pad =' + str(auto_pad) + ',strides =' + str(stride)+ ',dilations =' + str(dilations))
        inputShape = cases[0]['inputs'][0]['dims']
        #print('inputShape =' + str(inputShape))

        input = arr = np.array(cases[0]['inputs'][0]['data'], dtype='float32').reshape(inputShape)
        #print('input =' + str(input))

        filterShape = cases[0]['inputs'][1]['dims']
        #print('filterShape =' + str(filterShape))
        filter = arr = np.array(cases[0]['inputs'][1]['data'], dtype='float32').reshape(filterShape)
        #print('filter =' + str(filter))
        outputChannel = filterShape[0]
        #print('outputChannel =' + str(outputChannel))
        channel = inputShape[1]
        inputChannel = inputShape[1]
        #print("cases[0]['inputs'].length " + str(len(cases[0]['inputs'])))
        print('len of input: ' + str(len(cases[0]['inputs'])))
        if (len(cases[0]['inputs']) == 2):
            useBias = 0
            biasShape = [outputChannel]
            #bias = np.array([0, 0], dtype=np.float32).reshape(biasShape) 
            bias = np.zeros(biasShape).astype('f')
            print("bias =" + str(bias))
            print('biasShape =' + str(biasShape))
        else:
            useBias = 1
            biasShape = [outputChannel]
            # bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)
            bias = arr = np.array(cases[0]['inputs'][2]['data'], dtype='float32').reshape(biasShape)
            print("bias =" + str(bias))
            print('biasShape =' + str(biasShape))
        

        outputWebShape = cases[0]['outputs'][0]['dims']
        #print('outputWebShape =' + str(outputWebShape))

        outputWeb = arr = np.array(cases[0]['outputs'][0]['data'], dtype='float32').reshape(outputWebShape)
        #print('outputWeb =' + str(outputWeb))

        outOS = runOnnxScript()
        outPT = runPytorch()
        print(np.allclose(outOS.flatten().tolist(), outputWeb.flatten().tolist()))
        print(np.allclose(outPT.flatten().tolist(), outputWeb.flatten().tolist()))
        print(np.allclose(outOS, outPT))
        print(np.array_equiv(outOS, outPT))





#print((outOS))
#print((outPT))
#print((outputWeb))
#
#print((outOS.shape))
#print((outPT.shape))
#print((outputWeb.shape))
#
#print(type(outOS))
#print(type(outPT))
#print(type(outputWeb))

