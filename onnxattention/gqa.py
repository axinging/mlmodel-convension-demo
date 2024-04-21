# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 16
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
        xq, xk, xv = q, k, v

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        #xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        #self.cache_k = self.cache_k.to(xq)
        #self.cache_v = self.cache_v.to(xq)
#
        #self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        #self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = xk#self.cache_k[:bsz, : start_pos + seqlen]
        values = xv#self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return output


def test(qD, kD, vD, oD, inputDims, model_args): 
    model = Attention(model_args)
    q = torch.Tensor(qD).reshape(inputDims[0])
    k = torch.Tensor(kD).reshape(inputDims[1])
    v = torch.Tensor(vD).reshape(inputDims[2])
    print(model(q,k,v, 0, None))
    print(oD)

import json
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
            inputs.append((input["data"]))
            inputDims.append(input["dims"])

        inputCorrect = True
        for dim in inputDims:
            #print(dim[2], " dd ", inputDims[0][2])
            if (dim[1]!=inputDims[0][1]):
                inputCorrect = False
                print("Case skipped !!!!!!!!!!!!!! The seq len  of Q K V not match!")
                break
        #print("inputCorrect = " + str(inputCorrect))
        inputCorrect = True
        if (inputCorrect) :
            model_args: ModelArgs = ModelArgs(dim = inputDims[0][2], n_heads = num_heads, n_kv_heads = kv_num_heads)
            output = i["cases"][0]["outputs"][0]["data"]
            #print(str(model_args))
            #print(str(inputDims))
            test(inputs[0],inputs[1],inputs[2], output, inputDims, model_args)
            print("Case end !!!!!!!!!!!!!!")

   
    # Closing file
    f.close()


print(getCaseInputOutput("group-query-attention.jsonc"))
