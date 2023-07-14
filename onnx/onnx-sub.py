import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

t1 = np.array([4,8, 9]).astype(np.float32)
t2 = np.array([1,3, 9]).astype(np.float32)

# The required constants:
c1 = helper.make_node('Constant', inputs=[], outputs=['c1'], name='c1-node',
    value=helper.make_tensor(name='c1v', data_type=tp.FLOAT,
    dims=t1.shape, vals=t1.flatten()))
# The functional nodes:
n1 = helper.make_node('Sub', inputs=['a', 'b'], outputs=['output'], name='n1')
# Create the graph
g1 = helper.make_graph([n1], 'preprocessing',
 [helper.make_tensor_value_info('a', tp.FLOAT, [3]), helper.make_tensor_value_info('b', tp.FLOAT, [3])],
 [helper.make_tensor_value_info('output', tp.FLOAT, [3])])
# Create the model and check
m1 = helper.make_model(g1, producer_name='scailable-demo')
onnx.checker.check_model(m1)
# Save the model
MODEL_NAME = 'sub.onnx'
onnx.save(m1, MODEL_NAME)

import onnxruntime as ort
ort_sess = ort.InferenceSession(MODEL_NAME)
outputs = ort_sess.run(None, {'a': t1, 'b': t2})

# Print Result
print(outputs[0])
