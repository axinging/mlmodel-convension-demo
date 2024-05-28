# https://github.com/microsoft/onnxruntime/issues/1455

import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np
from collections import OrderedDict
import tempfile
from onnx import numpy_helper
import json

def createInputFile():
    t1 = np.array([4, 8, 9]).astype(np.float32)
    t2 = np.array([1, 3, 9]).astype(np.float32)
    fname1 = tempfile.mkstemp()[1]
    t1.tofile(fname1)
    fname2 = tempfile.mkstemp()[1]
    t2.tofile(fname2)
    return fname1, fname2

def getInputNodeShape(model):

    # iterate through inputs of the graph
    for input in model.graph.input:
        print (input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    print (d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                    print (d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print ("?", end=", ")  # unknown dimension with no name
        else:
            print ("unknown rank", end="")
        print()

def getUnknownNodeShape(model):
    from onnx import shape_inference
    inferred_model = shape_inference.infer_shapes(model)
    print("shape of unknown node: ")
    print(inferred_model.graph.value_info)
    print("shape of unknown!!!")

def traverseGraph(model):
    print("traverseGraph: ")
    # model is an onnx model
    graph = model.graph
    # graph inputs
    for input_name in graph.input:
        print(input_name)
    # graph parameters
    # for init in graph.init:
    #    print(init.name)
    # graph outputs
    for output_name in graph.output:
        print(output_name)
    # iterate over nodes
    '''
    for node in graph.node:
        # node inputs
        print('node in graph:')
        print(node)
        for idx, node_input_name in enumerate(node.input):
            print(idx, node_input_name)
        # node outputs
        for idx, node_output_name in enumerate(node.output):
            print(idx, node_output_name)
        
        print('node in graph end')
    '''
    print("traverseGraph end!!! ")

def saveObjectToJsonFile(dictionary):
    # Serializing json
    json_object = json.dumps(dictionary, indent=2)
    
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)

#ONNXFILE = 'full-pipeline.onnx'
ONNXFILE = './brainchop/model_5_channels.onnx'
# add all intermediate outputs to onnx net
ort_session = ort.InferenceSession(ONNXFILE)
org_outputs = [x.name for x in ort_session.get_outputs()]

model = onnx.load(ONNXFILE)

# add_value_info_for_constants(model)
# getInputNodeShape(model)
getUnknownNodeShape(model)
#getOutputNodeShape(model)

traverseGraph(model)


#https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

# https://github.com/microsoft/onnxruntime/issues/1455
for node in model.graph.node:
    # print("***********************node " + node.name)
    # print(node)
    # print("----------------------node end")
    # print("input")
    # for input in node.input:
    #     print("  " + input)
    # print("output")
    '''
    for output in node.output:
        if output not in org_outputs:
            print("  " + output)
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    '''
    for output in node.output:
        #print("  " + output)
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    #'''
            
print("----------------------- all node values")
'''
# excute onnx
ort_session = ort.InferenceSession(model.SerializeToString())
outputs = [x.name for x in ort_session.get_outputs()]
fname1, fname2 = createInputFile()
print(fname1 + fname2)
a = np.fromfile(fname1, dtype=np.float32)
b = np.fromfile(fname2, dtype=np.float32)

ort_outs = ort_session.run(outputs, {'a': a, 'b': b} ) 
#ort_outs = ort_session.run(outputs, {'x': a} )
# Try subgraph
# ort_outs = ort_session.run(outputs, {'output1': a, 'c1': b} )
ort_outs = OrderedDict(zip(outputs, ort_outs))
json_outs =  {'a': a.tolist(), 'b': b.tolist()}
for key, value in ort_outs.items():
    print(key, value)
    json_outs[key] = value.tolist()

json_data = json.dumps(json_outs)

print("JSON Data  ")
print(json_data)
saveObjectToJsonFile(json_data)
'''

'''
JSONFILE = 'dump.json'
import json


with open(JSONFILE) as f:
    d = json.load(f)

def parseNode(node):
    print(node)

for i in d['graph']['node']:
    parseNode(i)
'''
