import onnx
import json

def getInput(name):
    #model = onnx.load(r"jets-text-to-speech.onnx")
    model = onnx.load(name)

    # The model is represented as a protobuf structure and it can be accessed
    # using the standard python-for-protobuf methods

    # iterate through inputs of the graph
    for input in model.graph.input:
        print (input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        return str(input.type)
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            return tensor_type.shape.dim
            # iterate through dimensions of the shape:
            '''
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    print (d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                    print (d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print ("?", end=", ")  # unknown dimension with no name
            '''
        else:
            print ("unknown rank", end="")
        print()
        return ''


from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
app = Flask(__name__)

@app.route("/")
def home():
    print("---------------iii---home")
    return "hi"
@app.route("/index")
def index():
   print("---------------iii---")

@app.route('/login', methods=['GET', 'POST'])
def login():
   print("------------------")
   message = None
   if request.method == 'POST':
        datafromjs = request.form['name']
        result = getInput(datafromjs)
        resp = make_response('{"response": '+result+'}')
        print("------------------")
        resp.headers['Content-Type'] = "application/json"
        return resp
        return render_template('login.html', message='')

if __name__ == "__main__":
    app.run(debug = True)
