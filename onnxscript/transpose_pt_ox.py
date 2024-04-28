
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

"""
v = np.array([[[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8]],

         [[ 9, 10, 11, 12],
          [13, 14, 15, 16]]]], dtype=np.int32)
"""

def test(v):
    print("Before Transpose:")
    print(v)
    # (1, 2, 2, 4)
    print(v.shape)
    result1 = Transpose(v, perm)
    print("Transpose in ONNX 2:")
    print(result1.flatten())
    print(*result1.flatten(),sep=', ') 
    print(result1.shape)


v = np.array([2, 9, 4, 8, 12, 13, 14, 15], dtype=np.int32)
v = v.reshape([1,2,1,4])
#test(v)

#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
"""
v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120], dtype=np.int32)
v = v.reshape([2,3,4,5])
#test(v)

v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], dtype=np.int32)
v = v.reshape([1,2,4,5])
#test(v)


# 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
v = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56], dtype=np.int32)
v = v.reshape([1,2,2,4])
test(v)

v = np.array([40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55], dtype=np.int32)
v = v.reshape([1,2,2,4])
test(v)


v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
v = v.reshape([1,2,1,4])
test(v)


v = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
v = v.reshape([1,2,1,4])
test(v)
"""

v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
v = v.reshape([1,2,1,4])
test(v)

v = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
v = v.reshape([1,2,1,4])
test(v)


v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int32)
v = v.reshape([1,2,2,4])
test(v)

v = np.array([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15], dtype=np.int32)
v = v.reshape([1,2,2,4])
test(v)
