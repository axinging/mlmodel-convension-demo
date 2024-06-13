# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

#From:  https://github.com/meta-llama/llama/blob/main/llama/model.py#L304C9-L304C31
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import torch
import numpy as np
import onnx

# We use ONNX opset 15 to define the function below.
from onnxscript import FLOAT, script
from onnxscript import opset15 as op
import numpy as np

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np

import math
import random
import unittest

import numpy
import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper

@dataclass
class ModelArgs:
    dim: int = 16
    kv_dim: int = 16
    n_heads: int = 4
    n_kv_heads: Optional[int] = 2

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.kv_head_dim = args.kv_dim // args.n_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = q.shape
        _, kv_seqlen, _ = k.shape
        xq, xk, xv = q, k, v

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, kv_seqlen, self.n_local_kv_heads, self.kv_head_dim)
        xv = xv.view(bsz, kv_seqlen, self.n_local_kv_heads, self.kv_head_dim)

        #xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        #self.cache_k = self.cache_k.to(xq)
        #self.cache_v = self.cache_v.to(xq)
#
        #self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        #self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = xk#self.cache_k[:bsz, : start_pos + seqlen]
        values = xv#self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        ## BSNH
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        
        # TO BNSH
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        print("scores shape in llama " + str(scores.shape))
        print("output shape in llama " + str(output.shape))
        return output


# q_ro, k_cache_rep, v_cache_rep, None, new_mask, 0.0, None, causal=True, window_size=window_size
def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    #print(query_padding_mask)
    #print(key_padding_mask)
    #print(dropout_p)
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    print(" q shape = " + str(q.shape))
    print(" k shape = " + str(k.shape))
    print(" v shape = " + str(v.shape))
    print(" reorder_ops = " + str(reorder_ops))
    """
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    """
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    print(" k shape = " + str(k.shape))
    print(" v shape = " + str(v.shape))

    d = q.shape[-1]
    #print("d " + str(d))
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
        #print("scores .shape" + str(scores.shape))
        #print("scores " + str(scores))
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        NotImplemented
    if window_size[0] >= 0 or window_size[1] >= 0:
        NotImplemented
    print("scores shape in pt " + str(scores.shape))
    #return scores
    attention = torch.softmax(scores, dim=-1)
    print("attention  shape" + str(attention.shape))
    #print(attention)

    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        NotImplemented
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    #print("output shape in pt " + str(output.shape))
    return output#.to(dtype=dtype_og), attention.to(dtype=dtype_og)




def test(qD, kD, vD, oD, inputDims, model_args): 
    model = Attention(model_args)
    q = torch.Tensor(qD).reshape(inputDims[0])
    k = torch.Tensor(kD).reshape(inputDims[1])
    v = torch.Tensor(vD).reshape(inputDims[2])
    output = model(q,k,v, 0, None)
    #print('output from pytorch: ')
    #print(output)
    outputRef = torch.Tensor(oD).reshape(output.shape)
    print(torch.allclose(output, outputRef, rtol=1e-01, atol=1e-01,)) 
    #print(" output shape  in pt " + str(output.shape))
    #print('outputRef from outputRef: ')
    #print(outputRef)
    qShape = inputDims[0]
    kShape = inputDims[1] 
    vShape = inputDims[2]
    qShape = [qShape[0], qShape[1], model_args.n_heads, int(qShape[2]/model_args.n_heads)]
    kShape = [kShape[0], kShape[1], model_args.n_kv_heads, int(kShape[2]/model_args.n_kv_heads)]
    vShape = [vShape[0], vShape[1], model_args.n_kv_heads, int(vShape[2]/model_args.n_kv_heads)]
    outputLlama = attention_ref(q.reshape(qShape), k.reshape(kShape), v.reshape(vShape))
    #print('outputLlama from outputLlama: ')
    #print(outputLlama)
    outShape = output.shape
    #outShape = [outShape[0], outShape[1], model_args.n_heads, int(outShape[2]/model_args.n_heads)]

    #print(" output shape  " + str(output.shape))
    
    outputLlama = outputLlama.reshape(outShape)
    #print(" outputLlama shape  " + str(outputLlama.shape))
    #print(output)
    #print(outputRef)
    print(torch.allclose(output, outputLlama, rtol=1e-01, atol=1e-01,)) 

#import json
import json5 as json
def getCaseInputOutput(fileName):
    # Opening JSON file
    f = open(fileName)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    print(len(data))
    
    for i in data:
        #if (caseName == i["name"]):
        #print(i["attributes"][0])
        #print(i["attributes"][1]["kv_num_heads"])
        num_heads = 0
        kv_num_heads = 0
        for attr in i["attributes"]:
            print(attr)
            if (attr["name"] == "num_heads"):
                num_heads = attr["data"]
            if (attr["name"] == "kv_num_heads"):
                kv_num_heads = attr["data"]

        #print(num_heads, kv_num_heads)
        inputs = []
        inputDims = []
        #dim = 0
        for input in i["cases"][0]["inputs"]:
            if ("dims" in input and "data" in input):
                inputs.append((input["data"]))
                inputDims.append(input["dims"])

        inputCorrect = True
        '''
        for dim in inputDims:
            #print(dim[2], " dd ", inputDims[0][2])
            if (dim[1]!=inputDims[0][1]):
                inputCorrect = False
                print("Case skipped !!!!!!!!!!!!!! The seq len  of Q K V not match!")
                break
        '''
        #print("inputCorrect = " + str(inputCorrect))
        inputCorrect = True
        if (inputCorrect) :
            model_args: ModelArgs = ModelArgs(dim = inputDims[0][2], kv_dim = inputDims[1][2], n_heads = num_heads, n_kv_heads = kv_num_heads)
            output = i["cases"][0]["outputs"][0]["data"]
            #print(str(model_args))
            #print(str(inputDims))
            test(inputs[0],inputs[1],inputs[2], output, inputDims, model_args)
            print("Case end !!!!!!!!!!!!!!")
   
    # Closing file
    f.close()


print(getCaseInputOutput("group-query-attention-prompt.jsonc"))
