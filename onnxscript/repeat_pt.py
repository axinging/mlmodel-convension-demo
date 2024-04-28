
import torch
import numpy as np
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

keys0 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16]).reshape([1,2,2,4])
n_rep = 2
print(keys0.shape)
print(keys0)

keys = repeat_kv(keys0, n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
print(keys.shape)
print(keys0[0,0,1,0])
print(keys[0,0,3,0])
print(keys)


# Input array
arr = np.array([[1, 2], [3, 4],
                [5, 6], [7, 8]])

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16]).reshape([1,2,2,4])

# Creating repeated array using NumPy repeat() down the column
arr = np.repeat(a = arr, repeats = 2, axis=2)
print(arr)

b = np.array([[1, 2], [3, 4]])
np.tile(b, 2)


b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16]).reshape([1,2,2,4])
print(np.tile(b, 2).reshape([1,2,4,4]))
