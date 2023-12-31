from transformers import AutoTokenizer, CLIPTextModel
import os

import torch

print(torch.__version__)
proxy = os.environ.get('http_proxy1')
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

### Load Pytorch model

# https://huggingface.co/docs/transformers/model_doc/clip
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states

def foo(**d):
    return d

temp = foo(**inputs)
print(temp)
print(temp['input_ids'])
print(temp['attention_mask'])
print(type(model))
print(type(tokenizer))
print(type(inputs))

###  Pytorch2ONNX

pytorch_module = model

PATH_MODEL_ALL='torch_saved_model_cliptext.pt'
torch.save(pytorch_module, PATH_MODEL_ALL)
model = torch.load(PATH_MODEL_ALL)
model.eval()

dummy_input = torch.randn(10, 3, 224, 224, device="cpu")
ONNX_FILE = 'cliptext.onnx'
# Export the model
torch.onnx.export(model,               # model being run
                  foo(**inputs),                  # model input (or a tuple for multiple inputs)
                  ONNX_FILE,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

### ONNX2TF

import onnx
onnx_model = onnx.load(ONNX_FILE)
onnx.checker.check_model(onnx_model)
from onnx_tf.backend import prepare
tfmodel = prepare(onnx_model)  # run the loaded model
print("TFModel: ", type(tfmodel))
tfmodel.export_graph("onnx2tf_cliptext_data")
