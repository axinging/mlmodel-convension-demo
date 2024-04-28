import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import numpy as np


@script()
def Concat(X1, X2):
    """Hardmax is similar to ArgMax, with the result being encoded OneHot style."""
    return op.Concat(X1, X2, axis=0)


x1 = np.array([[[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8]],

         [[ 9, 10, 11, 12],
          [13, 14, 15, 16]]]], dtype=np.float32)


#print(Add(v, v))

print(x1.shape)
x2 = np.array([[ 101,  102,  103,  104],[105,  106,  107,  108]], dtype=np.float32)
print(x2.shape)
result = Concat(x1, x2)
print(result.shape)
print(result)


