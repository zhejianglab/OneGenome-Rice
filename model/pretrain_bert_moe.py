# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain BERT with Mixture of Experts (MoE)"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
import megatron.legacy.model
from megatron.core.models.bert.bert_model import BertModel
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module, ModuleSpec
from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec, bert_layer_local_spec
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core import mpu, tensor_parallel

# MOE imports
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, TEGroupedMLP, SequentialMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
        TEGroupedLinear,
    )
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    import warnings
    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_bert_moe_layer_with_transformer_engine_spec():
    """Use this spec to use lower-level Transformer Engine modules with MoE (required for fp8 training).

    Returns:
        ModuleSpec: Module specification with TE modules and MoE support
    """
    if not HAVE_TE:
        raise ImportError(
            "Transformer Engine is not installed. Please use local Bert layer spec instead."
        )

    # Define MoE submodules
    moe_submodules = MoESubmodules(
        experts=ModuleSpec(
            module=TEGroupedMLP,
            submodules=MLPSubmodules(
                linear_fc1=ModuleSpec(
                    module=TEGroupedLinear,
                    params={"parallel_mode": "column"}
                ),
                linear_fc2=ModuleSpec(
                    module=TEGroupedLinear,
                    params={"parallel_mode": "row"}
                ),
            ),
        ),
        shared_experts=ModuleSpec(
            module=SharedExpertMLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MoELayer,
                submodules=moe_submodules,
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_bert_moe_layer_local_spec():
    """Use this spec for an implementation using only modules in megatron core with MoE.

    Returns:
        ModuleSpec: Module specification with local modules and MoE support
    """
    # Define MoE submodules for local implementation
    moe_submodules = MoESubmodules(
        experts=ModuleSpec(
            module=SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=tensor_parallel.ColumnParallelLinear,
                linear_fc2=tensor_parallel.RowParallelLinear,
            ),
        ),
        shared_experts=ModuleSpec(
            module=SharedExpertMLP,
            submodules=MLPSubmodules(
                linear_fc1=tensor_parallel.ColumnParallelLinear,
                linear_fc2=tensor_parallel.RowParallelLinear,
            ),
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=tensor_parallel.ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=tensor_parallel.RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=ModuleSpec(
                module=MoELayer,
                submodules=moe_submodules,
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build the BERT model with MoE support."""

    print_rank_0('building BERT MoE model ...')

    args = get_args()
    config = core_transformer_config_from_args(args)
    num_tokentypes = 2 if args.bert_binary_head else 0

    # Validate MoE arguments
    if args.num_experts is None or args.num_experts == 0:
        raise ValueError("num_experts must be specified and greater than 0 for MoE BERT model")

    if args.use_legacy_models:
        raise ValueError("Legacy models do not support MoE. Please use Megatron-Core models.")
    
    # Select appropriate layer spec based on configuration
    if args.spec is None:
        if HAVE_TE:
            transformer_layer_spec = get_bert_moe_layer_with_transformer_engine_spec()
            print_rank_0('Using Transformer Engine spec for BERT MoE layers')
        else:
            transformer_layer_spec = get_bert_moe_layer_local_spec()
            print_rank_0('Using local spec for BERT MoE layers')
    elif args.spec[0] == 'local':
        print_rank_0('Using Local spec for BERT MoE transformer layers')
        transformer_layer_spec = get_bert_moe_layer_local_spec()
    else:
        transformer_layer_spec = import_module(args.spec)

    model = BertModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels',
            'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}
    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask,
                          tokentype_ids=types, lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    tokenizer = get_tokenizer()

    config = BERTMaskedWordPieceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        masking_probability=args.mask_prob,
        short_sequence_probability=args.short_seq_prob,
        masking_max_ngram=3,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=False,
        classification_head=args.bert_binary_head,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT MoE ...')

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        BERTMaskedWordPieceDataset,
        train_val_test_num_samples,
        lambda: mpu.get_tensor_model_parallel_rank() == 0,
        config,
    ).build()

    print_rank_0("> finished creating BERT MoE datasets ...")

    return train_ds, valid_ds, test_ds


def add_bert_moe_args(parser):
    """Add BERT MoE specific arguments."""
    group = parser.add_argument_group(title='BERT MoE')
    
    # MoE layer frequency - how often to use MoE vs dense layers
    group.add_argument('--bert-moe-layer-freq', type=int, default=1,
                       help='Frequency of MoE layers. 1 means all layers are MoE, '
                            '2 means every other layer is MoE, etc.')
    
    # Expert capacity and load balancing
    group.add_argument('--bert-moe-expert-capacity-factor', type=float, default=1.25,
                       help='Expert capacity factor for BERT MoE layers.')
    
    # Token dropping policy
    group.add_argument('--bert-moe-token-drop-policy', type=str, default='probs',
                       choices=['probs', 'position'],
                       help='Token dropping policy for BERT MoE.')
    
    return parser


if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Add BERT MoE specific arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_bert_moe_args(parser)

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, 
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase',
                           'num_experts': 8,  # Default number of experts
                           'moe_router_topk': 2,  # Default top-k routing
                           'moe_aux_loss_coeff': 0.01,  # Default aux loss coefficient
                           'moe_token_dispatcher_type': 'allgather',  # Default dispatcher
                           'expert_model_parallel_size': 1,  # Default expert parallelism
                           'moe_grouped_gemm': True,  # Enable grouped GEMM for better performance
                           'moe_router_load_balancing_type': 'aux_loss',  # Load balancing strategy
                           'moe_ffn_hidden_size': None,  # Use same as regular FFN by default
                           })