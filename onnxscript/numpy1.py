import numpy as np


input = np.array([[[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8]],

         [[ 9, 10, 11, 12],
          [13, 14, 15, 16]]]], dtype=np.float32)

print(input.shape)
shape = [1,2,2,4]
print(input * np.ones((2,2,2,4)))

t = input * np.ones((1,2,2, 2,4))
print(t.shape)
print(t)


import numpy as np

X = np.arange(8).reshape(4, 2)
y = np.arange(2).reshape(1, 2)  # create a 1x2 matrix
#print(X * y)
