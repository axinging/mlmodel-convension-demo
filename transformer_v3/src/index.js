import { pipeline } from '@xenova/transformers';

// Create feature extraction pipeline
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    device: 'webgpu',
    dtype: 'fp32', // or 'fp16'
});

// Generate embeddings
const sentences = ['That is a happy person', 'That is a very happy person'];
const output = await extractor(sentences, { pooling: 'mean', normalize: true });
console.log(output.tolist());
