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
    bias = np.array([100, 200], dtype=np.float32).reshape(biasShape)

    
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

buildAndRunBinaryGraph()
