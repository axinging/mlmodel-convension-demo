import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto# as tp
import numpy as np
from onnxruntime_extensions import get_library_path as _lib_path

node_def = helper.make_node(
    'MultiHeadAttention', # node name
    ['query','key','value'], # inputs
    ['output'], # outputs
    domain="com.microsoft",
)

num_heads_attr = helper.make_attribute("num_heads", 2)
kv_num_heads_attr = helper.make_attribute("kv_num_heads", 2)
node_def.attribute.append(num_heads_attr)
#node_def.attribute.append(kv_num_heads_attr)

queryInfo = helper.make_tensor_value_info('query', TensorProto.FLOAT, [1,2,4])
keyInfo = helper.make_tensor_value_info('key', TensorProto.FLOAT, [1,2,4])
valueInfo = helper.make_tensor_value_info('value', TensorProto.FLOAT, [1,2,4])
outputInfo = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,2,4])
graph_def = helper.make_graph(
    [node_def],
    "test-model",
    [queryInfo, keyInfo,valueInfo],
    [outputInfo],
)

model_def = helper.make_model(graph_def,
                              producer_name='onnx-example')
model_def.ir_version = 9
#onnx.checker.check_model(model_def)
# Save the model
MODEL_NAME = "test_model.onnx"
onnx.save(model_def, MODEL_NAME)

#so = ort.SessionOptions()
#so.register_custom_ops_library(_lib_path())
#sess = rt.InferenceSession("test_model.onnx", sess_options=so)

onnx.save(model_def, MODEL_NAME)
#ort_sess = ort.InferenceSession(MODEL_NAME, sess_options=so)
print("before seee")
ort_sess = ort.InferenceSession(MODEL_NAME, providers=["CUDAExecutionProvider"])

print(ort_sess.get_providers()) # output: ['CPUExecutionProvider']

#options = ort_sess.get_provider_options()
#option = options["CUDAExecutionProvider"]
#ort_sess.set_providers(["CUDAExecutionProvider"], [option])
#ort_sess.allow_released_opsets_only = False
query = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32).reshape((1,2,4))
key = np.array([1, 1, 1, 1, 2, 2, 2, 2]).astype(np.float32).reshape((1,2,4))
value = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32).reshape((1,2,4))
outputs = ort_sess.run(None, {'query': query, 'key':key, 'value': value})
print(outputs)

"""
[array([[[4.5718327, 5.5718327, 6.9718585, 7.9718585],
        [4.9983253, 5.9983253, 6.999901 , 7.999901 ]]], dtype=float32)]

        "inputs": [
          {
            "data": [1, 2, 3, 4, 5, 6, 7, 8],
            "dims": [1, 2, 4],
            "type": "float32"
          },
          {
            "data": [1, 1, 1, 1, 2, 2, 2, 2],
            "dims": [1, 2, 4],
            "type": "float32"
          },
          {
            "data": [1, 2, 3, 4, 5, 6, 7, 8],
            "dims": [1, 2, 4],
            "type": "float32"
          }
        ],
        "outputs": [
          {
            "data": [
              4.571832656860352, 5.571832656860352, 6.971858501434326, 7.971858501434326, 4.998325824737549,
              5.998325824737549, 6.999900817871094, 7.999900817871094
            ],
            "dims": [1, 2, 4],
            "type": "float32"
          }
        ]
"""