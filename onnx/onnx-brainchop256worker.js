import * as ort from "./web/dist/ort.all.mjs"

let feedsInfo = [];
ort.env.logLevel = 'verbose';
ort.env.debug = true;
function getFeedInfo(feed, type, data, dims) {
    const warmupTimes = 0;
    const runTimes = 1;
    for (i = 0; i < warmupTimes + runTimes; i++) {
        let typedArray;
        let typeBytes;
        if (type === 'bool') {
            data = [data];
            dims = [1];
            typeBytes = 1;
        } else if (type === 'int8') {
            typedArray = Int8Array;
        } else if (type === 'float16') {
            typedArray = Uint16Array;
        } else if (type === 'int32') {
            typedArray = Int32Array;
        } else if (type === 'uint32') {
            typedArray = Uint32Array;
        } else if (type === 'float32') {
            typedArray = Float32Array;
        } else if (type === 'int64') {
            typedArray = BigInt64Array;
        }
        if (typeBytes === undefined) {
            typeBytes = typedArray.BYTES_PER_ELEMENT;
        }

        let size, _data;
        if (Array.isArray(data) || ArrayBuffer.isView(data)) {
            size = data.length;
            _data = data;
        } else {
            size = dims.reduce((a, b) => a * b);
            if (data === 'random') {
                _data = typedArray.from({ length: size }, () => getRandom(type));
            } else {
                _data = typedArray.from({ length: size }, () => data);
            }
        }

        if (i > feedsInfo.length - 1) {
            feedsInfo.push(new Map());
        }
        feedsInfo[i].set(feed, [type, _data, dims, Math.ceil(size * typeBytes / 16) * 16]);
    }
    return feedsInfo;
}

async function loadJSON(url) {
    const res = await fetch(url);
    return await res.json();
}

const saveTemplateAsFile = (filename, dataObjToWrite) => {
    //let out = "[" + dataObjToWrite.map(el => JSON.stringify(el)).join(",") + "]";
    const blob = new Blob([JSON.stringify(dataObjToWrite)], { type: "text/json" });
    const link = document.createElement("a");

    link.download = filename;
    link.href = window.URL.createObjectURL(blob);
    link.dataset.downloadurl = ["text/json", link.download, link.href].join(":");

    const evt = new MouseEvent("click", {
        view: window,
        bubbles: true,
        cancelable: true,
    });

    link.dispatchEvent(evt);
    link.remove()
};

async function main() {
    try {
        // set option
        const option = {
            executionProviders: [
                {
                    name: 'webgpu',
                },
            ],
            graphOptimizationLevel: 'extended',
            optimizedModelFilePath1: 'opt.onnx'
        };
        // const session = await ort.InferenceSession.create('./brainchop/model_5_channels.onnx', option);
        const session = await ort.InferenceSession.create('./brainchop/model256.onnx', option);
        // const session = await ort.InferenceSession.create('./brainchop/model_21_channels_104classes.onnx', option);
        // model_21_channels_104classes
        const size = 256;
        const shape = [1, 1, size, size, size];
        // const temp = getFeedInfo("input", "float32", 0, shape);
        //console.log(temp);
        let dataA;// = temp[0].get('input')[1];
        let dataTemp = await loadJSON("./onnx-branchchop-input256.jsonc");
        dataA = dataTemp['data'];
        const tensorA = new ort.Tensor('float32', dataA, shape);
        const feeds = { "input": tensorA };
        // feed inputs and run
        console.log("before run");
        const results = await session.run(feeds);
        console.log("after run");
        // read from results
        const dataC = results.output.cpuData;//results[39].data;
        console.log(`data of result tensor 'c': ${dataC[0]},${dataC[1]},${dataC[2]},${dataC[10]},${dataC[100]},${dataC[1000]},${dataC[10000]},${dataC[100000]},${dataC[300000]}`);


        console.log(`data of result tensor 'c': ${dataC[dataC.length - 1]},${dataC[dataC.length - 2]},${dataC[dataC.length - 10]},${dataC[dataC.length - 100]},${dataC[dataC.length - 1000]},${dataC[dataC.length - 10000]},${dataC[dataC.length - 100000]},${dataC[dataC.length - 300000]}`);
    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }

    let i = 0;
    while(true) {
        i++;
        if(i %1000 ==0)
        console.log(i);
    }
}

main();