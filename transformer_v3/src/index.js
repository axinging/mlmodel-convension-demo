import {pipeline} from '@xenova/transformers';

// Create an Automatic Speech Recognition pipeline
const transcriber = await pipeline(
    'automatic-speech-recognition', 'Xenova/wav2vec2-bert-CV16-en', {
      device: 'wasm',
      dtype: 'fp32',  // or 'fp16'
    });

// Transcribe audio
const url =
    'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await transcriber(url);
// { text: 'and so my fellow americans ask not what your country can do for you
// ask what you can do for your country' }
console.log(output);
