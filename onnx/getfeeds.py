import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

model = onnx.load('padv19_float16.onnx')
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)
