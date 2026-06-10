# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os, torch, torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager

def add_arguments(parser):
    group = parser.add_argument_group(title='HybridDenseMoE_hf saver.')
    group.add_argument('--hf-tokenizer-path', type=str, default='/mnt/workspace/users/chenjh356/tokenizer/onehot_eod',
                        help='Huggingface tokenizer path. eg. /models/llama-2-hf/7b-chat.')


@contextmanager
def suspend_nn_inits():
    """
    create context manager for loading without init

    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  #saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  #replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def save_checkpoint(queue: mp.Queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    assert md.model_type == 'GPT'
    mag_conf = md.checkpoint_args
    torch_dtype = torch.float32

    # Import HybridDenseMoeConfig from the model file
    try:
        from hybrid_dense_moe_model import HybridDenseMoeConfig, HybridDenseMoeForCausalLM
    except ImportError:
        print("Error: Cannot import HybridDenseMoeConfig. Please ensure the model file is available.")
        exit(1)

    # Create layer types: first 3 layers are dense, rest are moe (based on training script)
    layer_types = ['dense'] * 3 + ['moe'] * (mag_conf.encoder_num_layers - 3)
    print(mag_conf)
    hybrid_conf = HybridDenseMoeConfig(
        vocab_size=mag_conf.padded_vocab_size,
        hidden_size=mag_conf.hidden_size,
        intermediate_size=mag_conf.ffn_hidden_size,
        num_hidden_layers=mag_conf.encoder_num_layers,
        num_attention_heads=mag_conf.num_attention_heads,
        num_key_value_heads=mag_conf.num_query_groups,
        max_position_embeddings=mag_conf.max_position_embeddings,
        rms_norm_eps=mag_conf.norm_epsilon,
        tie_word_embeddings=not mag_conf.untie_embeddings_and_output_weights,
        attention_bias=mag_conf.add_bias_linear,
        layer_types=layer_types,
        num_local_experts=8,  # Based on training script --num-experts 8
        moe_intermediate_size=8192,  # Based on training script --moe-ffn-hidden-size 8192
        num_experts_per_tok=2,  # Based on training script --moe-router-topk 2
        torch_dtype=torch_dtype
    )

    state_dict = {}
    def set_hf_param(name, tensor: torch.Tensor):
        weight_name = f'{name}.weight'
        state_dict[weight_name] = tensor.to(torch.float16)

    def set_hf_param_with_bias(name, tensor: torch.Tensor, bias_tensor: torch.Tensor = None):
        weight_name = f'{name}.weight'
        state_dict[weight_name] = tensor.to(torch.float16)
        if bias_tensor is not None:
            bias_name = f'{name}.bias'
            state_dict[bias_name] = bias_tensor.to(torch.float16)

    # Set embedding
    set_hf_param('model.embed_tokens', queue_get("embeddings")["word embeddings"])

    # Process each layer
    for i_layer in range(hybrid_conf.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f'model.layers.{i_layer}.'

        # Set LayerNorm weights
        set_hf_param(suffix + 'input_layernorm', message["input norm weight"])
        set_hf_param(suffix + 'post_attention_layernorm', message["post norm weight"])

        # Process attention weights - split QKV
        qkv_weight = message["qkv weight"]
        qkv_bias = message.get("qkv bias", None)

        # Split QKV weight
        qkv_weight = qkv_weight.view(hybrid_conf.num_key_value_heads, -1, hybrid_conf.hidden_size)
        qkv_weight = torch.split(qkv_weight, [
            hybrid_conf.hidden_size // hybrid_conf.num_key_value_heads,
            hybrid_conf.hidden_size // hybrid_conf.num_attention_heads,
            hybrid_conf.hidden_size // hybrid_conf.num_attention_heads,
        ], dim=1)

        # Split QKV bias if exists
        if qkv_bias is not None:
            qkv_bias = qkv_bias.view(hybrid_conf.num_key_value_heads, -1)
            qkv_bias = torch.split(qkv_bias, [
                hybrid_conf.hidden_size // hybrid_conf.num_key_value_heads,
                hybrid_conf.hidden_size // hybrid_conf.num_attention_heads,
                hybrid_conf.hidden_size // hybrid_conf.num_attention_heads,
            ], dim=1)

        # Set attention parameters
        set_hf_param_with_bias(suffix + 'self_attn.q_proj', qkv_weight[0].reshape(-1, hybrid_conf.hidden_size), qkv_bias[0] if qkv_bias else None)
        set_hf_param_with_bias(suffix + 'self_attn.k_proj', qkv_weight[1].reshape(-1, hybrid_conf.hidden_size), qkv_bias[1] if qkv_bias else None)
        set_hf_param_with_bias(suffix + 'self_attn.v_proj', qkv_weight[2].reshape(-1, hybrid_conf.hidden_size), qkv_bias[2] if qkv_bias else None)
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])

        # Process MLP based on layer type
        if layer_types[i_layer] == 'dense':
            # Dense layer - split linear_fc1 into gate_proj and up_proj
            linear_fc1_weight = message["mlp l0 weight W"]
            linear_fc1_bias = message.get("mlp l0 bias W", None)

            # Split the weight matrix
            gate_weight, up_weight = torch.split(linear_fc1_weight, linear_fc1_weight.size(0) // 2, dim=0)
            if linear_fc1_bias is not None:
                gate_bias, up_bias = torch.split(linear_fc1_bias, linear_fc1_bias.size(0) // 2, dim=0)
            else:
                gate_bias = up_bias = None

            set_hf_param_with_bias(suffix + 'mlp.gate_proj', gate_weight, gate_bias)
            set_hf_param_with_bias(suffix + 'mlp.up_proj', up_weight, up_bias)
            set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
        else:
            # MoE layer
            set_hf_param(suffix + 'mlp.gate', message["router weight"])

            # Process MoE experts - handle the grouped expert weights
            linear_fc1_weight = message["experts linear_fc1 weight"]
            linear_fc2_weight = message["experts linear_fc2 weight"]

            num_experts = 8
            expert_fc1_size = linear_fc1_weight.size(0) // num_experts
            expert_fc2_size = linear_fc2_weight.size(1) // num_experts

            for expert_idx in range(num_experts):
                # Extract expert weights
                fc1_start = expert_idx * expert_fc1_size
                fc1_end = (expert_idx + 1) * expert_fc1_size
                fc2_start = expert_idx * expert_fc2_size
                fc2_end = (expert_idx + 1) * expert_fc2_size

                expert_fc1_weight = linear_fc1_weight[fc1_start:fc1_end]
                expert_fc2_weight = linear_fc2_weight[:, fc2_start:fc2_end]

                # Split expert_fc1 into gate_proj and up_proj
                gate_weight, up_weight = torch.split(expert_fc1_weight, expert_fc1_weight.size(0) // 2, dim=0)

                set_hf_param(suffix + f'mlp.experts.{expert_idx}.gate_proj', gate_weight)
                set_hf_param(suffix + f'mlp.experts.{expert_idx}.up_proj', up_weight)
                set_hf_param(suffix + f'mlp.experts.{expert_idx}.down_proj', expert_fc2_weight)

    # Set final norm and output layer
    set_hf_param('model.norm', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    with suspend_nn_inits():
        print("Saving model to disk ...")
        model = HybridDenseMoeForCausalLM.from_pretrained(None, config=hybrid_conf, state_dict=state_dict, torch_dtype=torch_dtype)
        model.save_pretrained(args.save_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path)
    tokenizer.save_pretrained(args.save_dir)