# Some code here has been modified from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py
# --------------------------------------------------------

# Data-Juicer adopts Apache 2.0 license, the original license of this file
# is as follows:
# --------------------------------------------------------
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re

import torch
from modeling_megatron_llama import MegatronLlamaConfig
from transformers import AutoTokenizer
from transformers.modeling_utils import (
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    shard_checkpoint,
)


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers"
            " checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help="The maximum size for a checkpoint before being sharded. "
        "Checkpoints shard will then be each of size lower than this "
        "size. If expressed as a string, needs to be digits followed by "
        "a unit (like `5MB`). Only used when converting a Megatron "
        "checkpoint to a Transformers checkpoint.",
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "attention.dense": ".self_attn.o_proj.",
    "self_attention.dense": ".self_attn.o_proj.",
    # TODO: one to two vectors
    "mlp.dense_h_to_4h": ".mlp.{}_proj.",
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
}
transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
]


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken
    from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a
            nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to
    [num_splits * num_heads * hidden_size, :] for compatibility with later
    versions of NVIDIA Megatron-LM. The inverse operation is performed inside
    Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    If param is the weight tensor of the self-attention block, the returned
    tensor will have to be transposed one more time to be read by HuggingFace
    GPT2. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for
            (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective
    NVIDIA Megatron-LM checkpoint versions. Input is
    [num_splits * num_heads * hidden_size, :] and output is
    [num_heads * hidden_size * num_splits, :] for version 1.0 and
    [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If
    param is the weight tensor of the self-attention block, the param needs to
    be already transposed before calling this function.

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for
            (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the
    provided tensor parallel size, pipeline parallel size and pipeline parallel
    rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively
    add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers
    checkpoint. This handles Megatron checkpoints with different tensor
    parallelism and pipeline parallelism sizes. It saves the converted
    checkpoint into shards using HuggingFace Transformers checkpoint sharding
    functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: " f"{rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility "
            "only supports Megatron-LM checkpoints containing all the "
            "architecture, the tensor and pipeline model parallel size from "
            "the checkpoint instead of user having to manually specify all the"
            " details. Please save Megatron-LM checkpoint along with all the "
            "megatron arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    if megatron_args is not None:
        # dawei: use swish as activation function
        if megatron_args.swiglu:
            activation_function = "silu"
        elif megatron_args.bias_gelu_fusion:
            activation_function = "gelu_fast"
        elif megatron_args.openai_gelu:
            activation_function = "gelu_new"
        else:
            activation_function = "gelu"
    else:
        # in the very early days this used to be "gelu_new"
        activation_function = "gelu_new"
    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print("vocab size:", vocab_size)

    config = MegatronLlamaConfig(
        # dawei: from megatron-lm
        vocab_size=vocab_size,
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,  # 10880
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        hidden_act=activation_function,
        max_position_embeddings=megatron_args.max_position_embeddings,
        rms_norm_eps=megatron_args.layernorm_epsilon,
        # dawei: from official config of llama
        max_sequence_length=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        architectures=["MegatronLlamaForCausalLM"],
        use_bias=True,
    )

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    dtype = torch.float32
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the position embeddings.
    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )
    output_state_dict["model.embed_position.weight"] = position_embeddings.to(dtype)

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer.
        path = (
            "model.language_model.transformer"
            if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
            else "model.language_model.encoder"
        )
        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            # dawei: input_layernorm, self_attention, mlp, post_attention_layernorm  # noqa: E501
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                # dawei: input_layernorm.weight, input_layernorm.bias, self_attention.dense.bias,  # noqa: E501
                # dawei: self_attention_layernorm.weight, self_attention_layernorm.bias, mlp.dense_4h_to_h.bias  # noqa: E501
                # dawei: post_attention_layernorm.weight, post_attention_layernorm.bias  # noqa: E501
                params = val.to(dtype)
            else:
                # dawei: self_attention.query_key_value.weight, self_attention_query_value.bias, self_attention.dense.weight,  # noqa: E501
                #  mlp.dense_h_to_4h.weight, mlp.dense_h_to_4h.bias,
                #  mlp.dense_4h_to_h.weight
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h", "attention.dense"] else 0
                # dawei: maybe only stored in the first chunk

                # dawei: fix bug in swiglu and dense_h_to_4h.weight

                if op_name == "mlp.dense_h_to_4h" and weight_or_bias == "weight":
                    params_list = [val] + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ]
                    ws, vs = list(), list()
                    for p in params_list:
                        w, v = torch.chunk(p, 2, dim=0)
                        ws.append(w)
                        vs.append(v)
                    params = torch.cat(ws + vs, dim=dim).to(dtype)

                else:
                    params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=dim,
                    ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            # dawei: ignore the bias for layernorm
            if op_name.endswith("layernorm"):
                # dawei: input_layernorm & post_attention_layernorm
                if weight_or_bias == "weight":
                    # dawei: skip bias
                    ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                    output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

            # Transpose the QKV matrix.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "weight":
                # dawei: (gpt2) self_attention.query_key_value.weight
                out_val = megatron_to_transformers_fix_query_key_value_ordering(  # noqa: E501
                    params,
                    checkpoint_version,
                    3,
                    heads,
                    hidden_size_per_head,
                )
                # Megatron stores (3*D) x D but transformers-GPT2 expects
                # D x 3*D.

                # dawei: (3*D) x D
                out_val = out_val.contiguous()

                # dawei: split into 3 weight
                # (3*D) x D ==> D x D, still [out_dim, in_dim]
                q, k, v = torch.chunk(out_val, 3, dim=0)
                # Store.
                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v

            # Transpose the bias.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "bias":
                # dawei: (gpt2) self_attention.query_key_value.bias
                out_val = megatron_to_transformers_fix_query_key_value_ordering(  # noqa: E501
                    params, checkpoint_version, 3, heads, hidden_size_per_head
                )
                # dawei: split in to 3 bias
                q_b, k_b, v_b = torch.chunk(out_val, 3, dim=0)

                # Store. No change of shape.
                output_state_dict[layer_name + ".self_attn.q_proj.bias"] = q_b
                output_state_dict[layer_name + ".self_attn.k_proj.bias"] = k_b
                output_state_dict[layer_name + ".self_attn.v_proj.bias"] = v_b

            elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "weight":
                # dawei: mlp.dense_h_to_4h.weight
                out_name = megatron_to_transformers[op_name]
                gate, up = torch.chunk(params, 2, dim=0)
                output_state_dict[layer_name + out_name.format("gate") + "weight"] = gate
                output_state_dict[layer_name + out_name.format("up") + "weight"] = up

            # Transpose the weights.
            elif weight_or_bias == "weight":
                # dawei: self_attention.dense.weight, mlp.dense_4h_to_h.weight
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "weight"] = params

            elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "bias":
                # dawei: mlp.dense_h_to_4h.bias
                out_name = megatron_to_transformers[op_name]
                gate_b, up_b = torch.chunk(params, 2, dim=0)
                output_state_dict[layer_name + out_name.format("gate") + "bias"] = gate_b
                output_state_dict[layer_name + out_name.format("up") + "bias"] = up_b

            # Copy the bias.
            elif weight_or_bias == "bias":
                # dawei: (gpt2) self_attention.query_key_value.bias
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "bias"] = params

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found " f"{layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["model.norm.weight"] = params["final_layernorm.weight"].to(dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    output_state_dict["lm_head.weight"] = word_embeddings.to(dtype)

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)

    print("Tokenizer_name: ", args.tokenizer_name)
    if args.tokenizer_name is None:
        tokenizer_name = "gpt2"
    else:
        tokenizer_name = args.tokenizer_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Save tokenizer based on args
    if args.tokenizer_name is not None:
        print(f"Adding {tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in " f"{os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint "
            f"({args.max_shard_size}) and is going to be split in "
            f"{len(shards)} checkpoint shards. You can find where each "
            f"parameters has been saved in the index located at "
            f"{save_index_file}."
        )


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()

    convert_checkpoint_from_megatron_to_transformers(args)


if __name__ == "__main__":
    main()
