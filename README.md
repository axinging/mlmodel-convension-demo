## Test model file:

https://storage.googleapis.com/tfhub-modules/google/imagenet/mobilenet_v3_small_100_224/classification/5.tar.gz


## How to

pb to json: tensorflowjs_converter --input_format=tf_saved_model ./ ./predict_houses_tfjs
pb to pbtxt: pb2pbtext
