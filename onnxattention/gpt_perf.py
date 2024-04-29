import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
def generate_tokens(use_kv_cache):

    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=use_kv_cache)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with torch.no_grad():

        num_new_tokens = 500

        # tokenize the original input sentence
        inputs = gpt2_tokenizer("Hope is a", return_tensors="pt", add_special_tokens=False)

        start_time = time.time()
        gpt2.generate(**inputs, max_new_tokens=num_new_tokens, min_new_tokens=num_new_tokens)
        end_time = time.time()

        print(f"Time taken to generate {num_new_tokens} tokens: {end_time - start_time:.4f} seconds")
        print(f"Time taken per token: {(end_time - start_time)/num_new_tokens:.4f} seconds")


# measure latency with key-value caching disabled
print("Without key-value caching:")
generate_tokens(use_kv_cache=False)

# measure latency with key-value caching enabled
print("\nWith key-value caching:")
generate_tokens(use_kv_cache=True)