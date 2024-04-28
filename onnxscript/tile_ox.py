import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import numpy as np


@script()
def Expand(X, shape):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    return op.Expand(X, shape)

@script()
def Tile(X, Y):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    return op.Tile(X, Y)

v = np.array([[[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8]],

         [[ 9, 10, 11, 12],
          [13, 14, 15, 16]]]], dtype=np.float32)


v1 = np.array([1, 1,1,2], dtype=np.int64)



print(v.size)
print(v1.size)

# (1, 2, 2, 4)
print(v.shape)
result1 = Tile(v, v1).reshape([1,2,4,4])
print(result1)
print(result1.shape)

#print(Add(v, v))
"""
v2 = v # np.array([1])
print(v2.shape)
result = Tile(v2, op.Shape((1, 2, 2,2, 4)))
print(result.shape)
print(result)

"""

v1 = np.array([[1, 2], [3, 4]])
v2 = np.array([1, 2], dtype=np.int64)
result1 = Tile(v1, v2)
#print(result1)
#print(result1.shape)
