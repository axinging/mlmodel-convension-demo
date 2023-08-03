import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

from json import JSONEncoder
import numpy

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def numpyTypeAsString(dataType):
    if (dataType == np.int32):
        return "int32"
    elif (dataType == np.float32):
        return ""
    else:
        return ""

def createTensorsJson(inputs, outputs):
    inputArrayJson = []
    nameStr = ""
    for input in inputs:  # [::-1]:
        inputData = input if isinstance(input, numpy.ndarray) else str(input)
        inputArrayJson.insert(len(inputArrayJson), {
                              "data": inputData, "dims": input.shape, "type": str(input.dtype)})
        nameStr += "T["+str(input.shape).replace('(',
                                                 '').replace(')', '').rstrip(',') + "] "
    nameStr += "(" + str(inputs[0].dtype)+")"
    #outputJson = ({"data": output, "dims": output.shape,
    #              "type": str(output.dtype)})

    outputArrayJson = []    
    for output in outputs:  # [::-1]:
        outputData = output if isinstance(output, numpy.ndarray) else str(output)
        outputArrayJson.insert(len(outputArrayJson), {
                              "data": outputData, "dims": output.shape, "type": str(output.dtype)})
        nameStr += "T["+str(output.shape).replace('(',
                                                 '').replace(')', '').rstrip(',') + "] "
    return {"name": nameStr, "inputs": inputArrayJson, "outputs": outputArrayJson}


def createJsonFromTensors(testName, opName, caseInfos, suffix):
    cases = []
    for caseInfo in caseInfos:
        cases.insert(len(cases), createTensorsJson(
            caseInfo["inputs"], caseInfo["outputs"]))
    testJson = {"name": testName,  "operator": opName, "attributes": [], "cases": cases}
    # use dump() to write array into file
    encodedNumpyData = json.dumps([testJson], cls=NumpyArrayEncoder)

    # Writing to sample.json
    if (suffix != ""):
        suffix = "_" + suffix
    with open(opName.lower() + suffix + ".jsonc", "w") as outfile:
        outfile.write(encodedNumpyData)



def getNPType(dataType):
    if (dataType == tp.UINT32):
        return np.uint32
    elif (dataType == tp.INT32):
        return np.int32
    else:
        return np.float32

def genJsoncFomeOnnxFile(MODEL_NAME = "Sub_int_FLOAT.onnx"):
    DATA_TYPE = tp.FLOAT
    NP_TYPE = getNPType(DATA_TYPE)

    a = np.random.randn(3).astype(NP_TYPE)
    b = np.random.randn(3).astype(NP_TYPE)
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'a': a, 'b': b})
    # Print Result
    opName = "Sub"
    caseInfo1 = {"inputs": [a, b], "outputs": outputs}
    createJsonFromTensors(opName + " with no attributes", opName, [caseInfo1], numpyTypeAsString(NP_TYPE))

genJsoncFomeOnnxFile()
