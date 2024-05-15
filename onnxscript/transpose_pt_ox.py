
import torch
import numpy as np
import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import numpy as np

perm = [0, 2, 1, 3]

@script()
def Transpose(X):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    return op.Transpose(X, perm = [0, 2, 1, 3])

def transposeOne(v):

    # (1, 2, 2, 4)
    #print(v.shape)
    result1 = Transpose(v, perm)
    print("Transpose in ONNX 2:")
    print(*result1.flatten(),sep=', ') 
    return result1

import numpy as np
"""
shape= [1,7,2,4]
start = 1
v = np.array((np.arange(start, np.prod(shape)+start)), dtype=np.int32).reshape(shape)
print("before shape : " + str(shape))
print(v)
v = v.reshape(shape)
transposeOne(v)
"""
"""
shape= [1,7,2,4]
start = 0
v = np.array((np.arange(start, np.prod(shape)+start)), dtype=np.int32).reshape(shape)
print("Raw Data from 0 to: ")
print(*v.flatten(),sep=', ') 
print("Before Transpose:")
print(v)
#print(v)
v = v.reshape(shape)

v = transposeOne(v)
print("1st Transpose:")
print(v)

shape= [1,7,2,4]
start = 1
v = np.array((np.arange(start, np.prod(shape)+start)), dtype=np.int32).reshape(shape)
print("Raw Data from 1 to: ")
print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose:")
print(v)
"""


shape= [1,2,2,4]
start = 1
v = np.array([1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16], dtype=np.int32).reshape(shape)
#1, 9, 1, 1, 1, 12, 21, 131, 131, 22, 21, 2, 2, 2, 2, 2, 22, 21, 2, 2, 2, 131, 22, 21
#print("Raw Data from 1 to, BNSH: ")
#print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose, BSNH:")
print(v)




shape= [1,2,7,4]
v = np.array([              1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 47, 48,
              5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 37, 38, 39, 40, 49, 50, 51, 52, 53, 54, 55, 56], dtype=np.int32).reshape(shape)
#print("Raw Data from 1 to, BNSH: ")
#print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose, BSNH:")
print(v)


shape= [1,2,7,4]
v = np.array([1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36, 41, 42, 43, 44, 49, 50, 51, 52,
              5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 37, 38, 39, 40, 45, 46, 47, 48, 53, 54, 55, 56], dtype=np.int32).reshape(shape)
#print("Raw Data from 1 to, BNSH: ")
#print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose, BSNH:")
print(v)


shape= [1,7,2,4]
v = np.array([1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36, 41, 42, 43, 44, 49, 50, 51, 52,
              5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 37, 38, 39, 40, 45, 46, 47, 48, 53, 54, 55, 56], dtype=np.int32).reshape(shape)
#print("Raw Data from 1 to, BNSH: ")
#print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose, BSNH:")
print(v)


shape= [1,2,7,4]
v = np.array([1,2,3,4,17,18,19,20,33,34,35,36,49,50,51,52,13,14,15,16,29,30,31,32,45,46,47,48,9,10,11,12,25,26,27,28,41,42,43,44,5,6,7,8,21,22,23,24,37,38,39,40,53,54,55,56], dtype=np.int32).reshape(shape)
#print("Raw Data from 1 to, BNSH: ")
#print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose, BSNH:")
print(v)


shape= [1,2,7,4]
v = np.array([1,2,3,4,17,18,19,20,33,34,35,36,45,46,47,48,13,14,15,16,29,30,31,32,49,50,51,52,5,6,7,8,25,26,27,28,41,42,43,44,9,10,11,12,21,22,23,24,37,38,39,40,53,54,55,56], dtype=np.int32).reshape(shape)
#print("Raw Data from 1 to, BNSH: ")
#print(*v.flatten(),sep=', ') 
v = transposeOne(v)
print("1st Transpose, BSNH:")
print(v)
