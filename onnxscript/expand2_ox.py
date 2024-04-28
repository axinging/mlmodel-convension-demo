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
def Add(X, Y):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    return op.Add(X, Y)

v = np.array([[[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8]],

         [[ 9, 10, 11, 12],
          [13, 14, 15, 16]]]], dtype=np.float32)


"""
      {
        "name": "Expand 1 - float32",
        "inputs": [
          {
            "data": [1],
            "dims": [1, 1],
            "type": "float32"
          },
          {
            "data": [1, 4],
            "dims": [2],
            "type": "int64"
          }
        ],
        "outputs": [
          {
            "data": [1, 1, 1, 1],
            "dims": [1, 4],
            "type": "float32"
          }
        ]
      }
"""

# (1, 2, 2, 4)
print(v.shape)
result1 = Expand(v, op.Shape((1, 2, 4, 4)))
print(result1)
print(result1.shape)

#print(Add(v, v))

v2 = v # np.array([1])
print(v2.shape)
result = Expand(v2, op.Shape((1, 2, 2,2, 4)))
print(result.shape)
print(result)


