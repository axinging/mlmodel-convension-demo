import sys
import ort_test_dir_utils
import onnx_test_data_utils
import numpy as np


import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np
import onnx
from onnxconverter_common import float16

modelf32_path = 'pad_constant_f32_opset8.onnx'
modelf16_path = 'pad_constant_f16_opset8.onnx'
modelf32 = onnx.load(modelf32_path)
model_fp16 = float16.convert_float_to_float16(modelf32)
onnx.save(model_fp16, modelf16_path)


# example model with two float32 inputs called 'input1' (dims: {2, 1}) and 'input2' (dims: {'dynamic', 4})
model_path = modelf16_path
# when using the default data generation any symbolic dimension values must be provided
#symbolic_vals = {'dynamic':2} # provide value for symbolic dim named 'dynamic' in 'input2'

# let create_test_dir create random input in the (arbitrary) default range of -10 to 10.
# it will create data of the correct type based on the model.
ort_test_dir_utils.create_test_dir(model_path, 'examples', 'test1')

# alternatively some or all input can be provided directly. any missing inputs will have random data generated.
# symbolic dimension values are only required for input data that is randomly generated,
# so we don't need to provide that in this case as we're explicitly providing all inputs.
inputs = {'x': np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12,1, 2, 3, 4,5,6,7,8,9,10,11,12,
                1, 2, 3, 4,5,6,7,8,9,10,11,12,1, 2, 3, 4,5,6,7,8,9,10,11,12,1, 2, 3, 4,5,6,7,8,9,10,11,12]).reshape((1,3,4,5)).astype(np.float16)} # use 2 for the 'dynamic' dimension so shape is {2, 4}

ort_test_dir_utils.create_test_dir(model_path, 'examples', 'test2', name_input_map=inputs)

# can easily dump the input and output to visually check it's as expected
onnx_test_data_utils.dump_pb('examples/test2/test_data_set_0')
