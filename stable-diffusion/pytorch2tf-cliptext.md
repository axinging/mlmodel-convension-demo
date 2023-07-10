1. Load Pytorch model

```
# https://huggingface.co/docs/transformers/model_doc/clip
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
outputs = model(**inputs)
```
2.  Pytorch2ONNX
```
torch.onnx.export(model,  ..)
```

3. ONNX2TF

```
import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load(ONNX_FILE)
tfmodel = prepare(onnx_model)  # run the loaded model
tfmodel.export_graph("onnx2tf_cliptext_data")
```
