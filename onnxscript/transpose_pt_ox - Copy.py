
import torch
import numpy as np
import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import numpy as np

@script()
def Transpose(X):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    return op.Transpose(X, perm = [0, 2, 1, 3])

v = np.array([1, 12, 21, 131, 22, 21, 2, 2,131, 22, 21, 2, 2, 131, 22, 21], dtype=np.int32) #.reshape([1,2,8])

# (1, 2, 2, 4)
print(v.shape)
result1 = Transpose(v).reshape([1,2,2,4])
print("repeat in ONNX 2:")
print(result1)
print(result1.shape)
