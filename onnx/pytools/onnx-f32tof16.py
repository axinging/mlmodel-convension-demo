import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np
import onnx
from onnxconverter_common import float16

model = onnx.load("pad_constant_f32_opset8.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "pad_constant_f16_opset8.onnx")
