import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
'''
  TF:
  [batch, in_depth, in_height, in_width, in_channels].
  [filter_depth, filter_height, filter_width, in_channels,out_channels].



Pytorch:
Input: 
  (𝑁,𝐶𝑖𝑛,𝐷𝑖𝑛,𝐻𝑖𝑛,𝑊𝑖𝑛)(N,Cin,Din,Hin,Win) or (𝐶𝑖𝑛,𝐷𝑖𝑛,𝐻𝑖𝑛,𝑊𝑖𝑛)(Cin,Din,Hin,Win)
Filter:
  out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
  bias:
  out_channels
  Output: 
  (𝑁,𝐶𝑜𝑢𝑡,𝐷𝑜𝑢𝑡,𝐻𝑜𝑢𝑡,𝑊𝑜𝑢𝑡)(N,Cout,Dout,Hout,Wout) or (𝐶𝑜𝑢𝑡,𝐷𝑜𝑢𝑡,𝐻𝑜𝑢𝑡,𝑊𝑜𝑢𝑡)(Cout,Dout,Hout,Wout),


  ONNX:
  input:
   (N x C x D1 x D2 … x Dn

'''

#"conv3d, x=[1, 2, 1, 2, 1] f=[2, 1, 2, 1, 2] s=1 d=1 p=valid",
'''
torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
'''
def buildAndRunBinaryGraph():
    # NCDHW
    #  weight of size [2, 1, 2, 1, 2], expected input[1, 2, 1, 2, 1] to have 1 channels, but got 2 channels instead
    inputShape = [1, 2, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], dtype=np.float32).reshape(inputShape)
    # 1, 9, 1, 1, 1, 12, 21, 131, 131, 22, 21, 2, 2, 2, 2, 2, 22, 21, 2, 2, 2, 131, 22, 21
    # print("Raw Data from 1 to, BNSH: ")
    # print(*v.flatten(),sep=', ')
    # v = np.array([], dtype=np.float32).reshape(shape)
    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 2, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1,1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (2, 1,2)
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(output) #
    '''
    torch.Size([1, 2, 1, 1, 1])
tensor([[[[[ 6.6639]]], [[[15.6450]]]]]
    '''

def buildAndRunBinaryGraph2():
    # NCDHW
    inputShape = [1, 1, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([0.25, 0.5, 0.75, 1], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 1, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (2, 1,2)
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(output) #
    '''
    torch.Size([1, 2, 1, 1, 1])
tensor([[[[[ 6.6639]]], [[[15.6450]]]]]
    '''

def buildAndRunBinaryGraph3():
    # NCDHW
    inputShape = [1, 1, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 1, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (2, 1,2)
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(output) #
    '''
    torch.Size([1, 2, 1, 1, 1])
tensor([[[[[ 6.6639]]], [[[15.6450]]]]]
    '''


def buildAndRunBinaryGraph4():
    # NCDHW
    inputShape = [1, 1, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 1, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (2, 1,2)
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(output) #
    '''
    torch.Size([1, 2, 1, 1, 1])
tensor([[[[[ 6.6639]]], [[[15.6450]]]]]
    '''

def buildAndRunBinaryGraphInput():
    # NCDHW
    inputShape = [1, 2, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 2, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (filterShape[2], filterShape[3],filterShape[4])
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(output) #

def buildAndRunBinaryGraphC2():
    # NCDHW
    inputShape = [1, 2, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 2, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1,1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (filterShape[2], filterShape[3],filterShape[4])
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(repr(output.flatten())) #

def buildAndRunBinaryGraphC3():
    # NCDHW
    inputShape = [1, 3, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 3, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1,1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (filterShape[2], filterShape[3],filterShape[4])
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(repr(output.flatten())) #

# np.random.rand(3,2)
def buildAndRunBinaryGraphC4():
    # NCDHW
    inputShape = [1, 4, 2, 1, 2]
    inputChannel = inputShape[1]
    input = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4], dtype=np.float32).reshape(inputShape)

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, 4, 2, 1, 2]
    outChannel = filterShape[0]
    filter = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1,1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3,3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4], dtype=np.float32).reshape(filterShape)
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)

    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (filterShape[2], filterShape[3],filterShape[4])
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=1,dilation=1,padding='valid')
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print(repr(output.flatten())) #


'''
  {
    "name": "conv3d, x=[1, 2, 3, 1, 3] f=[1, 1, 1, 3, 3] s=1 d=1 p=valid",
    "operator": "Conv",
    "attributes": [
      { "name": "kernel_shape", "data": [2, 1, 2], "type": "ints" },
      { "name": "auto_pad", "data": "VALID", "type": "string" },
      { "name": "strides", "data": [1, 1, 1], "type": "ints" },
      { "name": "dilations", "data": [1, 1, 1], "type": "ints" }
    ],
    "cases": [
      {
        "name": "T[0]",
        "inputs": [
          {
            "data": [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
            "dims": [1, 2, 2, 1, 2],
            "type": "float32"
          },
          {
            "data": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2],
            "dims": [2, 2, 2, 1, 2],
            "type": "float32"
          }
        ],
        "outputs": [
          {
            "data": [6.375, 15.375],
            "dims": [1, 2, 1, 1, 1],
            "type": "float32"
          }
        ]
      }
    ]
  }
  '''
import json
def dumpAsJson(name, op, input, inputShape, filter, filterShape, kernel_size, output, outputShape, stride, dilation, padding, bias, biasShape):
    
    data = {}
    data['name'] = name + ', x=' + ''.join(str(inputShape))+ ', f=' + ''.join(str(filterShape)) + ', s='+str(stride)+ ', d='+str(dilation)+ ', p='+str(padding)
    data['operator'] = op
    data['attributes'] = [
      { "name": "kernel_shape", "data": kernel_size, "type": "ints" },
      { "name": "auto_pad", "data": padding, "type": "string" },
      { "name": "strides", "data": [1, 1, 1], "type": "ints" },
      { "name": "dilations", "data": [1, 1, 1], "type": "ints" }
    ]
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
    #data['operator'] = op
    json_data = json.dumps(data)
    print(json_data)
    with open(name+'.json', 'w') as f:
      json_data = json.dump(data, f)
      print(json_data)

# np.random.rand(3,2)
def buildAndRunBinaryGraphC5():
    # NCDHW
    channel = 3
    inputShape = [1, channel, 2, 1, 2]
    inputChannel = inputShape[1]
    #input = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4], dtype=np.float32).reshape(inputShape)
    input = (np.random.rand(*tuple(inputShape))*10).round(1).astype('f')
    print(input.flatten())

    # out_channels, in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2]
    filterShape = [2, channel, 2, 1, 2]
    outChannel = filterShape[0]
    #filter = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1,1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3,3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4], dtype=np.float32).reshape(filterShape)
    #filter = np.random.rand(*tuple(filterShape)).astype('f')\
    filter = (np.random.rand(*tuple(filterShape))*10).round(1).astype('f')
    print(filter.flatten())
    biasShape = [outChannel]
    bias = np.array([0, 0], dtype=np.float32).reshape(biasShape)
    
    #kernel_size = (filterShape[0], filterShape[1], filterShape[2])
    kernel_size = (filterShape[2], filterShape[3],filterShape[4])
    stride  = 1
    dilation = 1
    padding='valid'
    m = nn.Conv3d(inputChannel, outChannel, kernel_size, stride=stride,dilation=dilation,padding=padding)
    m.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    m.weight = nn.Parameter(torch.from_numpy(filter).float())
    # NCDHW
    #m = m.to(memory_format=torch.channels_last) 
    # non-square kernels and unequal stride and with padding
    output = m(torch.from_numpy(input))
    print(output.shape) #outShape = [1, 2, 1, 1, 1]
    print((output.flatten())) #
    dumpAsJson('conv3d', 'Conv', input.flatten().tolist(), inputShape, filter.flatten().tolist(), filterShape, kernel_size, output.flatten().tolist(), output.shape, stride, dilation, padding.upper(), bias, biasShape)

buildAndRunBinaryGraphC5()

#buildAndRunBinaryGraph4()
