# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
sys.path.append("./")
from io import open
import random
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from model.module_cross import CrossModel
from model.until_config import PretrainedConfig


from .utils import cached_path
import pdb

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}

devices=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_video_len=16,
        type_vocab_size=2,
        initializer_range=0.02,
        video_feature_size=1024,
        v_feature_size=2048,
        v_target_size=1601,
        v_hidden_size=768,
        v_num_hidden_layers=3,
        v_num_attention_heads=12,
        v_intermediate_size=3072,
        bi_hidden_size=1024,
        bi_num_attention_heads=16,
        co_num_layers=6,
        v_attention_probs_dropout_prob=0.1,
        v_hidden_act="gelu",
        v_hidden_dropout_prob=0.1,
        v_initializer_range=0.2,
        v_biattention_id=[0, 1],
        t_biattention_id=[10, 11],
        num_classes=1805,
        predict_feature=False,
        fast_mode=False,
        fixed_v_layer=0,
        fixed_t_layer=0,
        in_batch_pairs=False,
        fusion_method="mul",
        intra_gate=False,
        with_coattention=True
    ):

        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        assert len(v_biattention_id) == len(t_biattention_id)
        assert max(v_biattention_id) < v_num_hidden_layers
        assert max(t_biattention_id) < num_hidden_layers

        if isinstance(vocab_size_or_config_json_file, str) or (
            sys.version_info[0] == 2
            and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.max_video_len=max_video_len
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.v_feature_size = v_feature_size
            self.video_feature_size=video_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_hidden_layers = v_num_hidden_layers
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_biattention_id = v_biattention_id
            self.t_biattention_id = t_biattention_id
            self.v_target_size = v_target_size
            self.bi_hidden_size = bi_hidden_size
            self.bi_num_attention_heads = bi_num_attention_heads

            self.co_num_layers=co_num_layers
            self.num_classes=num_classes

            self.predict_feature = predict_feature
            self.fast_mode = fast_mode
            self.fixed_v_layer = fixed_v_layer
            self.fixed_t_layer = fixed_t_layer
            
            self.in_batch_pairs = in_batch_pairs
            self.fusion_method = fusion_method
            self.intra_gate = intra_gate
            self.with_coattention=with_coattention
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

try:
    from apex import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertImageAttention(nn.Module):
    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.v_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):
    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask=None, use_co_attention_mask=False):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1

        if use_co_attention_mask:
            attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow. 
        attention_scores2 = attention_scores2 + attention_mask2
        if use_co_attention_mask:
            attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        return context_layer1, context_layer2, (attention_probs1, attention_probs2)

class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):


        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2

class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask=None, use_co_attention_mask=False):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask, use_co_attention_mask
        )

        attention_output1, attention_output2 = self.biOutput(bi_output2, input_tensor1, bi_output1, input_tensor2)

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)
        
        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        self.num_hidden_layers=config.num_hidden_layers
        self.v_num_hidden_layers=config.v_num_hidden_layers

        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )

        self.v_layer = nn.ModuleList(
            [copy.deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)]
        )

        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))]
        )

        self.temperature=0.07

        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

    def info_nce_loss(self, features, batch_size,n_views):
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(devices)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(devices)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # print("negatives: ", negatives.size())

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(devices)

        logits = logits / self.temperature
        return logits, labels

    def forward(
            self,
            txt_embedding,
            entity_embedding,
            image_embedding,
            txt_attention_mask,
            image_attention_mask,
            entity_attention_mask,
            co_IT_attention_mask=None,
            co_IPV_attention_mask=None,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
        ):
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []
        all_encoder_layers_entity = []
        
        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_entity = []
        all_attention_IT_mask_c = []
        all_attention_IPV_mask_c = []
        
        _, num_regions, v_hidden_size = image_embedding.size()

        # sep encoder
        for idx in range(0, self.num_hidden_layers):
            image_embedding, image_attention_probs = self.v_layer[idx](image_embedding, image_attention_mask)
        
        for idx in range(0, self.v_num_hidden_layers):
            txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding, txt_attention_mask)

        for idx in range(0, self.v_num_hidden_layers):
            entity_embedding, entity_attention_probs = self.layer[idx](entity_embedding, entity_attention_mask)
        
        pooled_feature_v = self.v_pooler(image_embedding)
        pooled_feature_t = self.t_pooler(txt_embedding)
        pooled_feature_entity = self.t_pooler(entity_embedding)
        
        CLR_loss=0
        cnt=0
        batch_size = image_embedding.size(0)
        all_features=[pooled_feature_v,pooled_feature_t]
        for i in range(len(all_features)):
            for j in range(i+1,len(all_features)):
                features=torch.cat([all_features[i],all_features[j]],dim=0)
                logits, labels = self.info_nce_loss(features, batch_size, n_views=2)
                CLR_loss += self.loss_fct(logits, labels)
                cnt+=1
        cnt = 0
        all_features=[pooled_feature_v,pooled_feature_entity]
        for i in range(len(all_features)):
            for j in range(i+1,len(all_features)):
                features=torch.cat([all_features[i],all_features[j]],dim=0)
                logits, labels = self.info_nce_loss(features, batch_size, n_views=2)
                CLR_loss += self.loss_fct(logits, labels)
                cnt+=1
        CLR_loss/=cnt

        # cross transformer
        

        use_co_attention_mask = False
        count = 0
        for v_layer_id,t_layer_id in zip(self.v_biattention_id,self.t_biattention_id):
            # do the bi attention.
            image_embedding, entity_embedding, co_attention_probs = self.c_layer[count](
                image_embedding, image_attention_mask, entity_embedding, entity_attention_mask, co_IPV_attention_mask,
                use_co_attention_mask)

            # use_co_attention_mask = False
            if output_all_attention_masks:
                all_attention_IPV_mask_c.append(co_attention_probs)
            count += 1
            
        use_co_attention_mask = False
        count = 0
        for v_layer_id,t_layer_id in zip(self.v_biattention_id,self.t_biattention_id):
            # do the bi attention.
            image_embedding, txt_embedding, co_attention_probs = self.c_layer[count](
                image_embedding, image_attention_mask, txt_embedding, txt_attention_mask, co_IT_attention_mask,
                use_co_attention_mask)

            # use_co_attention_mask = False
            if output_all_attention_masks:
                all_attention_IT_mask_c.append(co_attention_probs)
            count += 1
            
        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)
            all_encoder_layers_entity.append(entity_embedding)
        return all_encoder_layers_t, all_encoder_layers_v, all_encoder_layers_entity, \
               CLR_loss, (all_attention_mask_t, all_attnetion_mask_v, all_attention_IT_mask_c, all_attention_IPV_mask_c)


class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertTextFc(nn.Module):
    def __init__(self, config):
        super(BertTextFc, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class BertEntityFc(nn.Module):
    def __init__(self, config):
        super(BertTextFc, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class BertImageFc(nn.Module):
    def __init__(self, config):
        super(BertImageFc, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size*2, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BertPreTrainingHeads_woITM(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads_woITM, self).__init__()
        self.t_fc = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.pv_fc = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.v_fc = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.imagePredictions = BertImagePredictionHead(config)

    def forward(
        self, sequence_output_t, sequence_output_v, sequence_output_pv
    ):
        sequence_output_t=self.t_fc(sequence_output_t)
        sequence_output_pv=self.pv_fc(sequence_output_pv)
        sequence_output_v=self.v_fc(sequence_output_v)
        prediction_scores_t = self.predictions(sequence_output_t)
        try:
            prediction_scores_pv = self.predictions(sequence_output_pv)
        except:
            pdb.set_trace()
        prediction_scores_v = self.imagePredictions(sequence_output_v)

        return prediction_scores_t, prediction_scores_v, prediction_scores_pv

class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token. bi_hidden_size
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)
        # self.decoder = nn.Linear(config.bi_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, default_gpu=True, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()

        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )

        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config,
        default_gpu=True,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        *inputs,
        **kwargs
    ):
        CONFIG_NAME = "bert_config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        TF_WEIGHTS_NAME = "model.ckpt"

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file,
                )
            )
            return None

        if default_gpu:
            if resolved_archive_file == archive_file:
                logger.info("loading archive file {}".format(archive_file))
            else:
                logger.info(
                    "loading archive file {} from cache at {}".format(
                        archive_file, resolved_archive_file
                    )
                )
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        elif resolved_archive_file[-3:] == 'bin':
            serialization_dir = '/'.join(resolved_archive_file.split('/')[:-1])
            WEIGHTS_NAME = resolved_archive_file.split('/')[-1]
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info(
                "extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir
                )
            )
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, tempdir)
            serialization_dir = tempdir
        # Load config
        # config_file = os.path.join(serialization_dir, CONFIG_NAME)
        # config = BertConfig.from_json_file(config_file)
        if default_gpu:
            logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(
                weights_path,
                map_location="cpu",
            )
            if 'state_dict' in dir(state_dict):
                state_dict = state_dict.state_dict()

        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(
            s.startswith("bert.") for s in state_dict.keys()
        ):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0 and default_gpu:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0 and default_gpu:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0 and default_gpu:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return model

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        # initilize embedding
        self.embeddings = BertEmbeddings(config)
        self.v_embeddings = BertImageEmbeddings(config)

        self.encoder = BertEncoder(config)
        # self.cross_config=PretrainedConfig.from_json_file("/data1/xl/multi_modal/CAPTURE/model/cross-base/cross_config_temp.json")
        self.cross_config = PretrainedConfig.from_json_file(
            "../../model/cross-base/cross_config.json")
        self.cross_config.num_hidden_layers=config.co_num_layers
        self.cross_encoder=CrossModel(self.cross_config)

        self.t_fc=BertTextFc(config)
        self.v_fc=BertImageFc(config)
        #self.entity_fc=BertEntityFc(config)

        self.apply(self.init_bert_weights)

    def _get_cross_output(self, sequence_output, visual_output,
                          attention_mask, image_mask):
        concat_features = torch.cat((sequence_output,visual_output),
                                    dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, image_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        image_type_ = torch.ones_like(image_mask)
        concat_type = torch.cat((text_type_,image_type_), dim=1)
        cross_output, pooled_output = self.cross_encoder(concat_features, concat_type, concat_mask)

        sequence_output_t,sequence_output_v = torch.split(cross_output,
                                         [attention_mask.size(-1),
                                          image_mask.size(-1)], dim=1)
        return sequence_output_t,sequence_output_v

    def forward(
            self,
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            input_entity=None,
            entity_segment_ids=None,
            entity_mask=None,
            each_entity_ids=None,
            each_entity_mask=None,
            each_entity_segment=None,
            co_IT_attention_mask=None,
            co_IPV_attention_mask=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
        ):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if entity_mask is None:
            entity_mask = torch.ones_like(input_entity)
        if each_entity_mask is None:
            each_entity_mask = torch.ones_like(each_entity_ids)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)


        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_entity_attention_mask = entity_mask.unsqueeze(1).unsqueeze(2)
        # extended_each_entity_attention_mask = each_entity_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        extended_entity_attention_mask = extended_entity_attention_mask.to(dtype=torch.float32)
        extended_entity_attention_mask = (1.0 - extended_entity_attention_mask) * -10000.0
        
        # extended_each_entity_attention_mask = extended_each_entity_attention_mask.to(dtype=torch.float32)
        # extended_each_entity_attention_mask = (1.0 - extended_each_entity_attention_mask) * -10000.0
        
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=torch.float32)
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_IT_attention_mask is None:
            co_IT_attention_mask = torch.zeros(input_txt.size(0), input_imgs.size(1), input_txt.size(1)).type_as(extended_image_attention_mask)

        if co_IPV_attention_mask is None:
            co_IPV_attention_mask = torch.zeros(input_entity.size(0), input_imgs.size(1), input_txt.size(1)).type_as(extended_image_attention_mask)
            
        extended_co_IT_attention_mask = co_IT_attention_mask.unsqueeze(1)
        extended_co_IT_attention_mask = extended_co_IT_attention_mask * 5.0
        extended_co_IT_attention_mask = extended_co_IT_attention_mask.to(dtype=torch.float32)
        
        extended_co_IPV_attention_mask = co_IPV_attention_mask.unsqueeze(1)
        extended_co_IPV_attention_mask = extended_co_IPV_attention_mask * 5.0
        extended_co_IPV_attention_mask = extended_co_IPV_attention_mask.to(dtype=torch.float32)

        embedding_T_output = self.embeddings(input_txt, token_type_ids)
        embedding_PV_output = self.embeddings(input_entity, entity_segment_ids)
        
        each_entity_output_t = torch.zeros([each_entity_ids.shape[0], 15, embedding_PV_output.shape[2]], device=each_entity_ids.device)
        for batch_i in range(each_entity_ids.shape[0]):
            each_entity_ids_i = each_entity_ids[batch_i,:,:].reshape(each_entity_ids.shape[1], each_entity_ids.shape[2])
            pairs_each_entity_segment_i = each_entity_segment[batch_i,:,:].reshape(each_entity_segment.shape[1], each_entity_segment.shape[2])
            each_entity_mask_i = each_entity_mask[batch_i,:,:].reshape(each_entity_mask.shape[1], each_entity_mask.shape[2])

            embedding_each_entity= self.embeddings(each_entity_ids_i, pairs_each_entity_segment_i)  #self.bert(each_entity_ids_i, each_entity_segment_i, each_entity_mask_i, output_all_encoded_layers=True)
            
            attention_mask_un = each_entity_mask_i.to(dtype=torch.float).unsqueeze(-1)
            attention_mask_un[:, 0, :] = 0.
            sequence_output = embedding_each_entity[-1] * attention_mask_un
            text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
            each_entity_output_t[batch_i,:,:] =  text_out #encoded_each_entity_layers[-1] #[batch, entity_num, 36, dim]
        
        v_embedding_output = self.v_embeddings(input_imgs.to(torch.float32), image_loc.to(torch.float32))

        # sep encoder
        encoded_layers_t,encoded_layers_v, encoded_layers_entity, \
        CLR_loss, all_attention_mask = self.encoder(
            embedding_T_output,
            embedding_PV_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_entity_attention_mask,
            extended_co_IT_attention_mask,
            extended_co_IPV_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]
        sequence_output_entity = encoded_layers_entity[-1]

        # co encoder
        sequence_output_t, sequence_output_v_1 = self._get_cross_output(
            sequence_output_t,sequence_output_v,
            attention_mask,image_attention_mask
        )
        
        sequence_output_entity, sequence_output_v_2 = self._get_cross_output(
            sequence_output_entity,sequence_output_v,
            entity_mask, image_attention_mask
        ) 
        
        sequence_output_v =  torch.cat((sequence_output_v_1, sequence_output_v_2), dim=2)
        # image_attention_mask = image_attention_mask.to(dtype=torch.float).unsqueeze(-1)
        # sequence_output_v = sequence_output_v * image_attention_mask
        # image_attention_mask_sum = torch.sum(image_attention_mask, dim=1, dtype=torch.float)
        # image_attention_mask_sum[image_attention_mask_sum == 0.] = 1.
        # sequence_output_v = torch.sum(sequence_output_v, dim=1) / image_attention_mask_sum
        
        sequence_output_t=self.t_fc(sequence_output_t)
        sequence_output_entity=self.t_fc(sequence_output_entity)
        sequence_output_v=self.v_fc(sequence_output_v)

        pooled_output_t=sequence_output_t[:,0]
        pooled_output_v=sequence_output_v[:,0]
        pooled_output_entity=sequence_output_entity[:,0]

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return sequence_output_t, sequence_output_v, sequence_output_entity, each_entity_output_t, \
               pooled_output_t, pooled_output_v, pooled_output_entity, \
               CLR_loss, all_attention_mask


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """
    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)        
        embeddings = self.LayerNorm(img_embeddings+loc_embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertVideoEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(BertVideoEmbeddings, self).__init__()

        self.video_embeddings = nn.Linear(config.video_feature_size, config.v_hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_video_len, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_videos):
        img_embeddings = self.video_embeddings(input_videos)

        seq_length = input_videos.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_videos.device
        )
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(img_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertForMultiModalPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiModalPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads_woITM(
            config, self.bert.embeddings.word_embeddings.weight
        )

        self.apply(self.init_bert_weights)
        self.predict_feature = config.predict_feature
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

        print("model's option for predict_feature is ", config.predict_feature)

        if self.predict_feature:
            self.vis_criterion = nn.MSELoss(reduction="none")
        else:
            self.vis_criterion = nn.KLDivLoss(reduction="none") 

    def forward(
            self,
            input_ids,
            image_feat,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            masked_lm_labels=None,
            image_label=None,
            image_target=None,
            next_sentence_label=None,
            entity_ids=None,
            entity_mask=None,
            entity_segment_ids=None,
            pairs_entity_token_labels=None,
            each_entity_ids=None,
            each_entity_mask=None,
            each_entity_segment=None,
            entity_index=None,
            entity_sim=None,
            output_all_attention_masks=False,
            return_features=False
        ):

        # in this model, we first embed the images.
        sequence_output_t,sequence_output_v, sequence_output_pv, each_entity_output_t, \
        pooled_output_t, pooled_output_v, pooled_output_pv, \
        CLR_loss, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            entity_ids,
            entity_segment_ids,
            entity_mask,
            each_entity_ids,
            each_entity_mask,
            each_entity_segment,
            co_IT_attention_mask=None,
            co_IPV_attention_mask=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
        )
        prediction_scores_t, prediction_scores_v, prediction_scores_pv = self.cls(
            sequence_output_t, sequence_output_v, sequence_output_pv
        )
        entity_contras_loss =  torch.tensor(0.).to(input_ids.device)
        entity_graph_contras_loss = torch.tensor(0.).to(input_ids.device)
        
        if(entity_sim is None):
            pass
        else:
            prob = random.random()
            entity_loss = nn.MarginRankingLoss()
            entity_contras_loss = torch.tensor(0.).to(input_ids.device)
            entity_graph_contras_loss = torch.tensor(0.).to(input_ids.device)
            entity_num = 0
            batch_index = []
            entity_fea = []
            for batch_i in range(each_entity_output_t.shape[0]):
                loc_index = (entity_index[batch_i,:,0]!=-2).nonzero(as_tuple=False)
                if(len(loc_index)==0):
                    continue
                batch_index.extend(entity_index[batch_i,loc_index,0])
                entity_fea.extend(each_entity_output_t[batch_i, loc_index, :].squeeze(1))
            batch_entity_sim = torch.tensor(entity_sim[torch.tensor(batch_index).cpu().type(torch.long),:]).cuda()
            batch_entity_sim = batch_entity_sim[:, torch.tensor(batch_index).cpu().type(torch.long)]
            _, index =  torch.sort(batch_entity_sim, descending=True)
            entity_fea_matrix = torch.zeros([len(entity_fea), entity_fea[0].shape[0]], dtype=torch.float32)
            for i in range(len(entity_fea)):
                try:
                    entity_fea_matrix[i,:] = entity_fea[i].clone().detach().requires_grad_(True)
                except:
                    pdb.set_trace()
            select_sample_num = 3
            graph_node_num = 5
            entity_graph_contras_loss = 0
        prediction_scores_v = prediction_scores_v[:, 1:]
        if self.predict_feature:
            image_target = image_target[:,1:]
            img_loss = self.vis_criterion(
                prediction_scores_v.reshape(prediction_scores_v.shape[0]*prediction_scores_v.shape[1], 1).to(torch.float32),
                image_target.reshape(image_target.shape[0]*image_target.shape[1], 1).to(torch.float32),
            )
            #img_loss = self.vis_criterion(prediction_scores_v, image_target)
            # 1 x 36 x 1601
            
            masked_img_loss = torch.sum(
                img_loss * (image_target == 1).unsqueeze(2).float().reshape(image_target.shape[0]*image_target.shape[1],1)
            ) / max(torch.sum((image_target == 1).unsqueeze(2).reshape(image_target.shape[0]*image_target.shape[1],1).expand_as(img_loss)),1)

        else:
            img_loss = self.vis_criterion(
                F.log_softmax(prediction_scores_v, dim=2), image_target
            )
            masked_img_loss = torch.sum(
                img_loss * (image_label == 1).unsqueeze(2).float()
            ) / max(torch.sum((image_label == 1)), 1)


        masked_lm_loss = self.loss_fct(
            prediction_scores_t.view(-1, self.config.vocab_size),
            masked_lm_labels.view(-1),
        )

        masked_pv_loss = self.loss_fct(
            prediction_scores_pv.view(-1, self.config.vocab_size),
            pairs_entity_token_labels.view(-1),
        )
        
        # total_loss = masked_lm_loss + next_sentence_loss + masked_img_loss
        if return_features:
            return masked_lm_loss.unsqueeze(0).to(torch.float32),\
                   masked_img_loss.unsqueeze(0).to(torch.float32),\
                   CLR_loss.unsqueeze(0).to(torch.float32),\
                   pooled_output_t,\
                   pooled_output_v 
        else:
            return masked_lm_loss.unsqueeze(0).to(torch.float32), \
                   masked_img_loss.unsqueeze(0).to(torch.float32),\
                   masked_pv_loss.unsqueeze(0).to(torch.float32),\
                   CLR_loss.unsqueeze(0).to(torch.float32), \
                   each_entity_output_t 
#.unsqueeze(0).to(torch.float32), \


class Capture_IT_ForClassification(BertPreTrainedModel):
    def __init__(self,config):
        super(Capture_IT_ForClassification,self).__init__(config)
        self.bert=BertModel(config)
        self.cls = BertPreTrainingHeads_woITM(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.dense1=nn.Linear(config.bi_hidden_size*2,config.bi_hidden_size)
        self.dropout=nn.Dropout(p=0.1)
        # num_classes=1797
        # num_classes=1805

        self.dense2=nn.Linear(config.bi_hidden_size,config.num_classes)
        self.apply(self.init_bert_weights)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            input_ids,
            image_feat,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            label=None
            ):

        sequence_output_t,sequence_output_v, \
        pooled_output_t, pooled_output_v, \
        CLR_loss, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False
        )
        pooled_output=torch.cat((pooled_output_t,
                                 pooled_output_v,),dim=-1)
        hidden_state=self.dense1(pooled_output)
        output_logits=self.dense2(self.dropout(hidden_state))

        if label!=None:
            classification_loss=self.loss_fct(output_logits,label)
            return classification_loss,output_logits,hidden_state
        else:
            return None,output_logits,hidden_state


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

