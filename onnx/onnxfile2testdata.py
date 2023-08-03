import sys
ONNX_ROOT = 'c:\\code-onnx\\share\\onnxruntime\\'
sys.path.append(ONNX_ROOT + '/tools/python')


import numpy
import numpy as np
import sys
import ort_test_dir_utils
import onnx_test_data_utils

# example model with two float32 inputs called 'input1' (dims: {2, 1}) and 'input2' (dims: {'dynamic', 4})
model_path = ONNX_ROOT + 'js\\test\\data\\node\\opset7\\test_add\model.onnx'
# C:\code-onnx\onnxruntime\js\test\data\node\opset7\test_add



# when using the default data generation any symbolic dimension values must be provided
symbolic_vals = {'dynamic':2} # provide value for symbolic dim named 'dynamic' in 'input2'

# let create_test_dir create random input in the (arbitrary) default range of -10 to 10.
# it will create data of the correct type based on the model.
ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test1', symbolic_dim_values_map=symbolic_vals)

# alternatively some or all input can be provided directly. any missing inputs will have random data generated.
# symbolic dimension values are only required for input data that is randomly generated,
# so we don't need to provide that in this case as we're explicitly providing all inputs.
inputs = {'x': np.random.randn(3, 4, 5).astype(np.int32),
          'y': np.random.randn(3, 4, 5).astype(np.int32)} # use 2 for the 'dynamic' dimension so shape is {2, 4}

ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test2', name_input_map=inputs)

# can easily dump the input and output to visually check it's as expected
onnx_test_data_utils.dump_pb('temp/examples/test2/test_data_set_0')


# Check if results are correct.
try:
    ort_test_dir_utils.run_test_dir('temp/examples/test1')
    ort_test_dir_utils.run_test_dir('temp/examples/test2/expand_elimination.onnx')
except Exception:
    print("Exception:", sys.exc_info()[1])
