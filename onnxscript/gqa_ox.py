from onnxscript import FLOAT, script
from onnxscript import opset18 as op
import onnxscript

import onnx
from onnx import helper as onnx_helper
import dataclasses
from onnxscript.rewriter import function_rule

# Code new pattern with onnxscript.
#op = onnxscript.opset18
msft_op = onnxscript.values.Opset("com.microsoft", 1)
# Workaround onnxscript error by specifying the output shape here.



@dataclasses.dataclass
class AttnSizeConfig:
    num_attention_heads: int
    num_key_value_heads: int
    head_size: int
    hidden_size: int


def infer_attn_size_config() -> AttnSizeConfig:
    head_size = 4 #present_value_ir.shape[3]
    num_key_value_heads = 2# present_value_ir.shape[1]
    hidden_size = 4 #attn_output_ir.shape[2]
    num_attention_heads = hidden_size // head_size
    return AttnSizeConfig(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_size=head_size,
        hidden_size=hidden_size,
    )

# Infer size configurations from the function.
attn_size_config = infer_attn_size_config(function)
cos_sin_gather_size = [attn_size_config.head_size // 2]

def gqa(
    hidden_states,
    position_id,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    cos_cached,
    sin_cached,
    o_proj_weight,
):
    q = op.MatMul(hidden_states, op.Transpose(q_proj_weight, [1, 0]))
    k = op.MatMul(hidden_states, op.Transpose(k_proj_weight, [1, 0]))
    v = op.MatMul(hidden_states, op.Transpose(v_proj_weight, [1, 0]))
    # NOTE: Depending on transformers version, the shape of cos/sin is different.
    # In later version, the shape is [seq_len, head_size], so the Squeeze is not needed.
    # In this version, the shape is [1, 1, seq_len, head_size], hence the below Squeeze.
    cos = op.Slice(op.Squeeze(cos_cached, [0, 1]), [0], cos_sin_gather_size, [1])
    sin = op.Slice(op.Squeeze(sin_cached, [0, 1]), [0], cos_sin_gather_size, [1])
    q_rope = msft_op.RotaryEmbedding(q, position_id, cos, sin, interleaved=False)
    k_rope = msft_op.RotaryEmbedding(k, position_id, cos, sin, interleaved=False)
    batch_size = op.Slice(op.Shape(hidden_states), [0], [1], [0])
    sequence_length = op.Slice(op.Shape(hidden_states), [1], [2], [0])
    past_seq_lengths = op.ConstantOfShape(
        batch_size,
        value=onnx_helper.make_tensor(
            "past_seq_lengths", onnx.TensorProto.INT32, [1], [0]
        ),
    )
    total_seq_lengths = op.Cast(sequence_length, to=onnx.TensorProto.INT32)
    gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
        q_rope,
        k_rope,
        v,
        None,
        None,
        past_seq_lengths,
        total_seq_lengths,
        kv_num_heads=attn_size_config.num_key_value_heads,
        num_heads=attn_size_config.num_attention_heads,
    )
    attn_output = op.MatMul(gqa_output, op.Transpose(o_proj_weight, [1, 0]))
    return present_value, present_key, attn_output

def gqaattn(
    q_rope,
    k_rope,
    v,
    k_proj_weight,
    v_proj_weight,
    cos_cached,
    sin_cached,
    o_proj_weight,
):
    gqa_output, present_key, present_value = msft_op.GroupQueryAttention(
        q_rope,
        k_rope,
        v,
        None,
        None,
        None,
        None,
        kv_num_heads=attn_size_config.num_key_value_heads,
        num_heads=attn_size_config.num_attention_heads,
    )
    return gqa_output


@script()
def gqaattnscript(
    q_rope,
    k_rope,
    v,
    k_proj_weight,
    v_proj_weight,
    cos_cached,
    sin_cached,
    o_proj_weight,
):
    return msft_op.GroupQueryAttention(
        q_rope,
        k_rope,
        v,
        None,
        None,
        None,
        None,
        kv_num_heads=attn_size_config.num_key_value_heads,
        num_heads=attn_size_config.num_attention_heads,
    )

import numpy as np
v = np.array([[[[ 1,  2,  3,  4],
          [ 5,  6,  7,  8]],

         [[ 9, 10, 11, 12],
          [13, 14, 15, 16]]]], dtype=np.float32)

q_rope = v
k_rope =v
print(gqaattnscript(q_rope, k_rope, v))
