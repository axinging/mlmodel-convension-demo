# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

#From:  https://github.com/meta-llama/llama/blob/main/llama/model.py#L304C9-L304C31
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


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

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    print(x.shape[1])
    print(x.shape[-1])
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

max_batch_size: int = 32
max_seq_len: int = 2048

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
        self.cache_k = torch.zeros(
            (
                max_batch_size,
                max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

        self.cache_v = torch.zeros(
            (
                max_batch_size,
                max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
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

        print("before embedding xk.shape = " + str(xk.shape))
        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        print("xk.shape = " + str(xk.shape))
        #print("self.cache_k.shape = " + str(self.cache_k.shape))


        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
#
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        print("xk.shape = " + str(xk.shape))
        #print("self.cache_k.shape = " + str(self.cache_k.shape))


        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        print("keys.shape = " + str(keys.shape))

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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def test(qD, kD, vD, oD, inputDims, model_args, start_pos): 
    model = Attention(model_args)
    q = torch.Tensor(qD).reshape(inputDims[0])
    k = torch.Tensor(kD).reshape(inputDims[1])
    v = torch.Tensor(vD).reshape(inputDims[2])
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads, 3)
    #model_args.dim // model_args.n_heads, max_seq_len * 2)
    print(freqs_cis.shape)
    output = model(q,k,v, start_pos, freqs_cis, None)
    outputRef = torch.Tensor(oD).reshape(output.shape)
    print(torch.allclose(output, outputRef, rtol=1e-01, atol=1e-01,)) 
    print(output)
    print(outputRef)

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
            model_args: ModelArgs = ModelArgs(dim = inputDims[0][2], kv_dim = inputDims[1][2], n_heads = num_heads, n_kv_heads = kv_num_heads)
            output = i["cases"][0]["outputs"][0]["data"]
            #print(str(model_args))
            #print(str(inputDims))
            test(inputs[0],inputs[1],inputs[2], output, inputDims, model_args, 0)
            test(inputs[0],inputs[1],inputs[2], output, inputDims, model_args, 1)
            print("Case end !!!!!!!!!!!!!!")

   
    # Closing file
    f.close()


print(getCaseInputOutput("group-query-attention.jsonc"))
