import onnx

def getInput(modelPath):
    #model = onnx.load(r"jets-text-to-speech.onnx")
    model = onnx.load(modelPath)

    # The model is represented as a protobuf structure and it can be accessed
    # using the standard python-for-protobuf methods

    # iterate through inputs of the graph
    inputInfo = {}
    for input in model.graph.input:
        # get type of input tensor
        tensor_type = input.type.tensor_type
        inputInfo[input.name] = input.type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            return tensor_type.shape.dim
        else:
            print ("unknown rank", end="")
    return inputInfo

def getOutput():
    #model = onnx.load(r"jets-text-to-speech.onnx")
    model = onnx.load(r"./models/whisper-tiny-decoder-no-edit.onnx")
    #print(model.graph.output)

    # The model is represented as a protobuf structure and it can be accessed
    # using the standard python-for-protobuf methods

    # iterate through inputs of the graph
    for input in model.graph.output:
        print (input.name)
        # get type of input tensor
        tensor_type = input.type.tensor_type
        #return ''
inputInfo = getInput(r"./models/op_test_generated_model_Where_with_no_attributes.onnx")
print(inputInfo)
