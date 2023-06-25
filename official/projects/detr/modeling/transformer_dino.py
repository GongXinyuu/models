# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Specialized Transformers for DETR.

the position embeddings are added to the query and key for every self- and
cross-attention layer.
"""
import math
from typing import List, Dict, Tuple, Union, Optional  # noqa
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling import models


def refpoint_initializer(shape, dtype=None):
  weights = tf.random.normal(shape, mean=0., stddev=1., dtype=dtype)

  # Uniformly initialize the first two columns
  weights[:, :2] = tf.random.uniform((shape[0], 2), minval=0, maxval=1, dtype=dtype)

  # Apply inverse_sigmoid function
  weights[:, :2] = inverse_sigmoid(weights[:, :2])

  # Set the first two columns to not require gradient updates
  weights[:, :2] = tf.stop_gradient(weights[:, :2])

  return weights


class FocalBiasInitializer(tf.keras.initializers.Initializer):
  def __init__(self, prior_prob=0.01):
    self.bias_value = -math.log((1 - prior_prob) / prior_prob)

  def __call__(self, shape, dtype=None):
    return tf.ones(shape, dtype=dtype) * self.bias_value


class FanInBiasInitializer(tf.keras.initializers.Initializer):
  def __call__(self, shape, dtype=None):
    fan_in = shape[-1]
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return tf.keras.initializers.RandomUniform(-bound, bound)(shape, dtype)


def inverse_sigmoid(x, eps=1e-3):
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    x1 = tf.clip_by_value(x, clip_value_min=eps, clip_value_max=float('inf'))
    x2 = tf.clip_by_value(1 - x, clip_value_min=eps, clip_value_max=float('inf'))
    return tf.math.log(x1 / x2)


class BBoxEmbed(tf.keras.Model):
  def __init__(self, hidden_size, prefix: str="dino"):
    super(BBoxEmbed, self).__init__()
    self.dense_0 = tf.keras.layers.Dense(
      hidden_size, activation="relu",
      kernel_initializer=tf.keras.initializers.HeUniform(),
      bias_initializer=FanInBiasInitializer(),
      name=f"{prefix}/box_dense_0")
    self.dense_1 = tf.keras.layers.Dense(
      hidden_size, activation="relu",
      kernel_initializer=tf.keras.initializers.HeUniform(),
      bias_initializer=FanInBiasInitializer(),
      name=f"{prefix}/box_dense_1")
    self.dense_2 = tf.keras.layers.Dense(
      4,  kernel_initializer=tf.keras.initializers.Zeros(),
      bias_initializer=tf.keras.initializers.Zeros(),
      name=f"{prefix}/box_dense_2")

  def call(self, inputs):
    x = self.dense_0(inputs)
    x = self.dense_1(x)
    x = self.dense_2(x)
    return x


def gen_sineembed_for_position(pos_tensor):
  # pos_tensor: [batch_size, n_query, 4]
  scale = 2 * math.pi
  dim_t = tf.range(128, dtype=tf.float32)
  dim_t = 10000 ** (2 * (dim_t // 2) / 128)
  x_embed = pos_tensor[:, :, 0] * scale
  y_embed = pos_tensor[:, :, 1] * scale
  pos_x = x_embed[:, :, tf.newaxis] / dim_t
  pos_y = y_embed[:, :, tf.newaxis] / dim_t

  pos_x_shape = tf.shape(pos_x)
  pos_y_shape = tf.shape(pos_y)

  pos_x = tf.reshape(tf.stack((tf.sin(pos_x[:, :, 0::2]), tf.cos(pos_x[:, :, 1::2])), axis=3),
                     shape=(pos_x_shape[0], pos_x_shape[1], 128))
  pos_y = tf.reshape(tf.stack((tf.sin(pos_y[:, :, 0::2]), tf.cos(pos_y[:, :, 1::2])), axis=3),
                     shape=(pos_y_shape[0], pos_y_shape[1], 128))

  if pos_tensor.shape[-1] == 2:
    pos = tf.concat((pos_y, pos_x), axis=2)
  elif pos_tensor.shape[-1] == 4:
    w_embed = pos_tensor[:, :, 2] * scale
    pos_w = w_embed[:, :, tf.newaxis] / dim_t

    pos_w_shape = tf.shape(pos_w)

    pos_w = tf.reshape(tf.stack((tf.sin(pos_w[:, :, 0::2]), tf.cos(pos_w[:, :, 1::2])), axis=3),
                       shape=(pos_w_shape[0], pos_w_shape[1], 128))

    h_embed = pos_tensor[:, :, 3] * scale
    pos_h = h_embed[:, :, tf.newaxis] / dim_t

    pos_h_shape = tf.shape(pos_h)

    pos_h = tf.reshape(tf.stack((tf.sin(pos_h[:, :, 0::2]), tf.cos(pos_h[:, :, 1::2])), axis=3),
                       shape=(pos_h_shape[0], pos_h_shape[1], 128))

    pos = tf.concat((pos_y, pos_x, pos_w, pos_h), axis=2)
  else:
    raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.shape[-1]))

  return pos

def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shape, learnedwh=None):
  """
  Input:
      - memory: bs, \sum{hw}, d_model
      - memory_padding_mask: bs, h, w
      - spatial_shapes: nlevel, 2
      - learnedwh: 2
  Output:
      - output_memory: bs, \sum{hw}, d_model
      - output_proposals: bs, \sum{hw}, 4
  """
  N_ = tf.shape(memory)[0]
  base_scale = 4.0
  proposals = []
  lvl = 3  # TODO: check
  # for lvl, (H_, W_) in enumerate(spatial_shapes):
  # H_, W_ = spatial_shape
  H_ = tf.shape(memory_padding_mask)[1]
  W_ = tf.shape(memory_padding_mask)[2]
  # mask_flatten_ = tf.reshape(memory_padding_mask[:, :(H_ * W_)], (N_, H_, W_, 1))
  mask_flatten_ = tf.expand_dims(memory_padding_mask, axis=-1)  # N_, H_, W_, 1
  mask_flatten_bool = tf.cast(mask_flatten_, tf.bool)
  counter_mask_flatten_bool = ~mask_flatten_bool

  valid_H = tf.reduce_sum(tf.cast(~counter_mask_flatten_bool[:, :, 0, 0], tf.float32), axis=1)
  valid_W = tf.reduce_sum(tf.cast(~counter_mask_flatten_bool[:, 0, :, 0], tf.float32), axis=1)

  grid_y, grid_x = tf.meshgrid(tf.linspace(0, H_ - 1, H_), tf.linspace(0, W_ - 1, W_))
  grid = tf.stack([grid_x, grid_y], axis=-1)  # H_, W_, 2
  grid = tf.cast(grid, tf.float32)

  scale = tf.reshape(tf.stack([valid_W[..., tf.newaxis], valid_H[..., tf.newaxis]], axis=1), (N_, 1, 1, 2))
  grid = (tf.expand_dims(grid, 0) + 0.5) / scale

  if learnedwh is not None:
    wh = tf.ones_like(grid) * tf.math.sigmoid(learnedwh) * (2.0 ** lvl)
  else:
    wh = tf.ones_like(grid) * 0.05 * (2.0 ** lvl)

  proposal = tf.reshape(tf.concat((grid, wh), axis=-1), (N_, -1, 4))
  proposals.append(proposal)

  output_proposals = tf.concat(proposals, axis=1)
  output_proposals_valid = tf.reduce_all((output_proposals > 0.01) & (output_proposals < 0.99), axis=-1,
                                         keepdims=True)
  output_proposals = tf.math.log(output_proposals / (1 - output_proposals))  # unsigmoid
  memory_padding_mask_squeeze_bool = tf.cast(tf.reshape(memory_padding_mask, [N_, -1]), tf.bool)
  counter_memory_padding_mask_squeeze_bool= ~memory_padding_mask_squeeze_bool
  output_proposals = tf.where(tf.expand_dims(counter_memory_padding_mask_squeeze_bool, -1), float('inf'), output_proposals)
  output_proposals = tf.where(~output_proposals_valid, float('inf'), output_proposals)

  output_memory = memory
  output_memory = tf.where(tf.expand_dims(counter_memory_padding_mask_squeeze_bool, -1), 0., output_memory)
  output_memory = tf.where(~output_proposals_valid, 0., output_memory)

  return output_memory, output_proposals


class MLP(tf.keras.Model):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
    super().__init__()
    self.num_layers = num_layers
    h = [hidden_dim] * (num_layers - 1)
    self.layers_list = [tf.keras.layers.Dense(k,  kernel_initializer=tf.keras.initializers.HeUniform(),
      bias_initializer=FanInBiasInitializer()) for n, k in zip([input_dim] + h, h + [output_dim])]

  def call(self, x):
    for i, layer in enumerate(self.layers_list):
      x = tf.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    return x

class TransformerEncoder(tf.keras.layers.Layer):
  """Transformer encoder.

  Transformer encoder is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self,
               num_layers=6,
               num_attention_heads=8,
               intermediate_size=2048,
               activation="relu",
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-6,
               intermediate_dropout=0.0,
               **kwargs):
    """Initialize a Transformer encoder.

    Args:
      num_layers: Number of layers.
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate (Feedforward) layer.
      activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability.
      attention_dropout_rate: Dropout probability for attention layers.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      **kwargs: key word arguemnts passed to tf.keras.layers.Layer.
    """

    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.num_attention_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout

  def build(self, input_shape):
    """Implements build() for the layer."""
    self.encoder_layers = []
    for i in range(self.num_layers):
      self.encoder_layers.append(
        TransformerEncoderBlock(
          num_attention_heads=self.num_attention_heads,
          inner_dim=self._intermediate_size,
          inner_activation=self._activation,
          output_dropout=self._dropout_rate,
          attention_dropout=self._attention_dropout_rate,
          use_bias=self._use_bias,
          norm_first=self._norm_first,
          norm_epsilon=self._norm_epsilon,
          inner_dropout=self._intermediate_dropout,
          attention_initializer=tf_utils.clone_initializer(
            models.seq2seq_transformer.attention_initializer(
              input_shape[2])),
          name=("layer_%d" % i)))
    self.output_normalization = tf.keras.layers.LayerNormalization(
      epsilon=self._norm_epsilon, dtype="float32")
    super(TransformerEncoder, self).build(input_shape)

  def get_config(self):
    config = {
      "num_layers": self.num_layers,
      "num_attention_heads": self.num_attention_heads,
      "intermediate_size": self._intermediate_size,
      "activation": self._activation,
      "dropout_rate": self._dropout_rate,
      "attention_dropout_rate": self._attention_dropout_rate,
      "use_bias": self._use_bias,
      "norm_first": self._norm_first,
      "norm_epsilon": self._norm_epsilon,
      "intermediate_dropout": self._intermediate_dropout
    }
    base_config = super(TransformerEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, encoder_inputs, attention_mask=None, pos_embed=None):
    """Return the output of the encoder.

    Args:
      encoder_inputs: A tensor with shape `(batch_size, input_length,
        hidden_size)`.
      attention_mask: A mask for the encoder self-attention layer with shape
        `(batch_size, input_length, input_length)`.
      pos_embed: Position embedding to add to every encoder layer.

    Returns:
      Output of encoder which is a `float32` tensor with shape
        `(batch_size, input_length, hidden_size)`.
    """
    for layer_idx in range(self.num_layers):
      encoder_inputs = self.encoder_layers[layer_idx](
        [encoder_inputs, attention_mask, pos_embed])

    output_tensor = encoder_inputs
    output_tensor = self.output_normalization(output_tensor)

    return output_tensor

class TransformerEncoderBlock(tf.keras.layers.Layer):
  """TransformerEncoderBlock layer.

  This layer implements the Transformer Encoder from
  "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
  which combines a `tf.keras.layers.MultiHeadAttention` layer with a
  two-layer feedforward network. The only difference: position embedding is
  added to the query and key of self-attention.

  References:
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [BERT: Pre-training of Deep Bidirectional Transformers for Language
     Understanding](https://arxiv.org/abs/1810.04805)
  """

  def __init__(self,
               num_attention_heads,
               inner_dim,
               inner_activation: str,
               output_range=None,
               kernel_initializer="he_uniform",
               bias_initializer=FanInBiasInitializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               inner_dropout=0.0,
               attention_initializer=None,
               attention_axes=None,
               **kwargs):
    """Initializes `TransformerEncoderBlock`.

    Args:
      num_attention_heads: Number of attention heads.
      inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network.
      inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network.
      output_range: the sequence output range, [0, output_range) for slicing the
        target sequence. `None` means the target sequence is not sliced.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: Dropout probability for within the attention layer.
      inner_dropout: Dropout probability for the first Dense layer in a
        two-layer feedforward network.
      attention_initializer: Initializer for kernels of attention layers. If set
        `None`, attention layers use kernel_initializer as initializer for
        kernel.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      **kwargs: keyword arguments/
    """
    super().__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._attention_dropout = attention_dropout
    self._attention_dropout_rate = attention_dropout
    self._output_dropout = output_dropout
    self._output_dropout_rate = output_dropout
    self._output_range = output_range
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._inner_dropout = inner_dropout
    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
        attention_initializer)
    else:
      self._attention_initializer = tf_utils.clone_initializer(
        self._kernel_initializer)
    self._attention_axes = attention_axes

  def build(self, input_shape):
    if isinstance(input_shape, tf.TensorShape):
      input_tensor_shape = input_shape
    elif isinstance(input_shape, (list, tuple)):
      input_tensor_shape = tf.TensorShape(input_shape[0])
    else:
      raise ValueError(
        "The type of input shape argument is not supported, got: %s" %
        type(input_shape))
    einsum_equation = "abc,cd->abd"
    if len(input_tensor_shape.as_list()) > 3:
      einsum_equation = "...bc,cd->...bd"
    hidden_size = input_tensor_shape[-1]
    if hidden_size % self._num_heads != 0:
      raise ValueError(
        "The input size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_head_size = int(hidden_size // self._num_heads)
    common_kwargs = dict(
      kernel_regularizer=self._kernel_regularizer,
      bias_regularizer=self._bias_regularizer,
      activity_regularizer=self._activity_regularizer,
      kernel_constraint=self._kernel_constraint,
      bias_constraint=self._bias_constraint)
    self._attention_layer = tf.keras.layers.MultiHeadAttention(
      num_heads=self._num_heads,
      key_dim=self._attention_head_size,
      dropout=self._attention_dropout,
      use_bias=self._use_bias,
      kernel_initializer=self._attention_initializer,
      bias_initializer=self._bias_initializer,
      attention_axes=self._attention_axes,
      name="self_attention",
      **common_kwargs)
    self._attention_dropout = tf.keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
      tf.keras.layers.LayerNormalization(
        name="self_attention_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32))
    self._intermediate_dense = tf.keras.layers.EinsumDense(
      einsum_equation,
      output_shape=(None, self._inner_dim),
      bias_axes="d",
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      name="intermediate",
      **common_kwargs)
    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      # TODO(b/154538392): Investigate this.
      policy = tf.float32
    if self._inner_activation != "prelu":
      self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._inner_activation, dtype=policy)
    else:
      self._intermediate_activation_layer = tf.keras.layers.PReLU(
        alpha_initializer=tf.keras.initializers.Constant(0.25), dtype=policy)
    self._inner_dropout_layer = tf.keras.layers.Dropout(
      rate=self._inner_dropout)
    self._output_dense = tf.keras.layers.EinsumDense(
      einsum_equation,
      output_shape=(None, hidden_size),
      bias_axes="d",
      name="output",
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      **common_kwargs)
    self._output_dropout = tf.keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
      name="output_layer_norm",
      axis=-1,
      epsilon=self._norm_epsilon,
      dtype=tf.float32)

    super(TransformerEncoderBlock, self).build(input_shape)

  def get_config(self):
    config = {
      "num_attention_heads": self._num_heads,
      "inner_dim": self._inner_dim,
      "inner_activation": self._inner_activation,
      "output_dropout": self._output_dropout_rate,
      "attention_dropout": self._attention_dropout_rate,
      "output_range": self._output_range,
      "kernel_initializer": tf_utils.serialize_initializer(
        self._kernel_initializer, use_legacy_format=True
      ),
      "bias_initializer": tf_utils.serialize_initializer(
        self._bias_initializer, use_legacy_format=True
      ),
      "kernel_regularizer": tf_utils.serialize_regularizer(
        self._kernel_regularizer, use_legacy_format=True
      ),
      "bias_regularizer": tf_utils.serialize_regularizer(
        self._bias_regularizer, use_legacy_format=True
      ),
      "activity_regularizer": tf_utils.serialize_regularizer(
        self._activity_regularizer, use_legacy_format=True
      ),
      "kernel_constraint": tf_utils.serialize_constraint(
        self._kernel_constraint, use_legacy_format=True
      ),
      "bias_constraint": tf_utils.serialize_constraint(
        self._bias_constraint, use_legacy_format=True
      ),
      "use_bias": self._use_bias,
      "norm_first": self._norm_first,
      "norm_epsilon": self._norm_epsilon,
      "inner_dropout": self._inner_dropout,
      "attention_initializer": tf_utils.serialize_initializer(
        self._attention_initializer, use_legacy_format=True
      ),
      "attention_axes": self._attention_axes,
    }
    base_config = super(TransformerEncoderBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Transformer self-attention encoder block call.

    Args:
      inputs: a single tensor or a list of tensors. `input tensor` as the single
        sequence of embeddings. [`input tensor`, `attention mask`] to have the
        additional attention mask. [`input tensor`, `attention mask`, `query
        embed`] to have an additional position embedding to add.

    Returns:
      An output tensor with the same dimensions as input/query tensor.
    """
    input_tensor, attention_mask, pos_embed = inputs

    key_value = None

    if self._output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:self._output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor[:, 0:self._output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:self._output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor
    attention_output = self._attention_layer(
      query=target_tensor + pos_embed,
      key=key_value + pos_embed,
      value=key_value,
      attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)
    if self._norm_first:
      attention_output = source_tensor + attention_output
    else:
      attention_output = self._attention_layer_norm(target_tensor +
                                                    attention_output)
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      return source_attention_output + layer_output

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    return self._output_layer_norm(layer_output + attention_output)


class TransformerDecoder(tf.keras.layers.Layer):
  """Transformer decoder.

  Like the encoder, the decoder is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self,
               num_layers=6,
               num_attention_heads=8,
               intermediate_size=2048,
               activation="relu",
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               use_bias=True,
               norm_epsilon=1e-6,
               intermediate_dropout=0.0,
               query_dim=4,
               modulate_hw_attn=True,
               iter_update=True,
               **kwargs):
    """Initialize a Transformer decoder.

    Args:
      num_layers: Number of layers.
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate (Feedforward) layer.
      activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability.
      attention_dropout_rate: Dropout probability for attention layers.
      use_bias: Whether to enable use_bias in attention layer. If set `False`,
        use_bias in attention layer is disabled.
      norm_epsilon: Epsilon value to initialize normalization layers.
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      **kwargs: key word arguemnts passed to tf.keras.layers.Layer.
    """
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.num_attention_heads = num_attention_heads
    self.query_dim = query_dim
    self.modulate_hw_attn = modulate_hw_attn
    self.iter_update = iter_update

    self._intermediate_size = intermediate_size
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._use_bias = use_bias
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout

  def build(self, input_shape):
    """Implements build() for the layer."""
    target_tensor_shape = tf.TensorShape(input_shape)
    if len(target_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerLayer expects a three-dimensional input of "
                       "shape [batch, sequence, width].")
    hidden_size = target_tensor_shape[2]
    if hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self.num_attention_heads))
    self.hidden_size = hidden_size
    self.attention_head_size = int(hidden_size) // self.num_attention_heads
    self.bbox_embed = BBoxEmbed(hidden_size, prefix="decoder")

    self.query_scale = MLP(hidden_size, hidden_size, hidden_size, 2)
    self.ref_point_head = MLP(self.query_dim // 2 * hidden_size, hidden_size, hidden_size, 2)
    if self.modulate_hw_attn:
      self.ref_anchor_head = MLP(hidden_size, hidden_size, 2, 2)

    self.decoder_layers = []
    for i in range(self.num_layers):
      self.decoder_layers.append(
          TransformerDecoderBlock(
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self._intermediate_size,
              intermediate_activation=self._activation,
              dropout_rate=self._dropout_rate,
              attention_dropout_rate=self._attention_dropout_rate,
              use_bias=self._use_bias,
              norm_epsilon=self._norm_epsilon,
              intermediate_dropout=self._intermediate_dropout,
              attention_initializer=tf_utils.clone_initializer(
                  models.seq2seq_transformer.attention_initializer(
                      input_shape[2])),
              is_first=(i == 0),
              name=("layer_%d" % i)))
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=self._norm_epsilon, dtype="float32")
    super(TransformerDecoder, self).build(input_shape)

  def get_config(self):
    config = {
        "num_layers": self.num_layers,
        "num_attention_heads": self.num_attention_heads,
        "query_dim": self.query_dim,
        "modulate_hw_attn": self.modulate_hw_attn,
        "iter_update": self.iter_update,
        "intermediate_size": self._intermediate_size,
        "activation": self._activation,
        "dropout_rate": self._dropout_rate,
        "attention_dropout_rate": self._attention_dropout_rate,
        "use_bias": self._use_bias,
        "norm_epsilon": self._norm_epsilon,
        "intermediate_dropout": self._intermediate_dropout
    }
    base_config = super(TransformerDecoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           target,
           memory,
           self_attention_mask=None,
           cross_attention_mask=None,
           cache=None,
           decode_loop_step=None,
           return_all_decoder_outputs=False,
           memory_pos_embed=None,
           refpoints_unsigmoid=None, # bs, num_queries, 4
           ):
    """Return the output of the decoder layer stacks.

    Args:
      target: A tensor with shape `(batch_size, target_length, hidden_size)`.
      memory: A tensor with shape `(batch_size, input_length, hidden_size)`.
      self_attention_mask: A tensor with shape `(batch_size, target_len,
        target_length)`, the mask for decoder self-attention layer.
      cross_attention_mask: A tensor with shape `(batch_size, target_length,
        input_length)` which is the mask for encoder-decoder attention layer.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
        {layer_n: {"k": A tensor with shape `(batch_size, i, key_channels)`,
                   "v": A tensor with shape `(batch_size, i, value_channels)`},
                     ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.
      return_all_decoder_outputs: Return all decoder layer outputs. Note that
        the outputs are layer normed. This is useful when introducing per layer
        auxiliary loss.
      refpoints_unsigmoid: A tensor that is added to the query and key of the
        self-attention layer.
      memory_pos_embed: A tensor that is added to the query and key of the
        cross-attention layer.

    Returns:
      Output of decoder.
      float32 tensor with shape `(batch_size, target_length, hidden_size`).
    """

    output_tensor = target
    decoder_outputs = []
    reference_points = tf.sigmoid(refpoints_unsigmoid)  # bs, num_queries, 4
    ref_points = [reference_points]

    for layer_idx in range(self.num_layers):
      obj_center = reference_points[..., :self.query_dim]  # [batch_size, num_queries, 4]
      # get sine embedding for the query vector
      query_sine_embed = gen_sineembed_for_position(obj_center)
      query_pos = self.ref_point_head(query_sine_embed)  # [batch_size, num_queries, hidden_size]

      # For the first decoder layer, we do not apply transformation over p_s
      if layer_idx == 0:
        pos_transformation = 1
      else:
        pos_transformation = self.query_scale(output_tensor)

      # apply transformation
      query_sine_embed = query_sine_embed[..., :self.hidden_size] * pos_transformation

      if self.modulate_hw_attn:
        refHW_cond = tf.sigmoid(self.ref_anchor_head(output_tensor))  # bs, nq, 4
        query_sine_embed = tf.concat([
          query_sine_embed[..., :self.hidden_size // 2] * tf.expand_dims(refHW_cond[..., 1] / obj_center[..., 3], axis=-1),
          query_sine_embed[..., self.hidden_size // 2:] * tf.expand_dims(refHW_cond[..., 0] / obj_center[..., 2], axis=-1),
        ], axis=-1)

      transformer_inputs = [
          output_tensor, memory, cross_attention_mask, self_attention_mask,
          query_pos, memory_pos_embed, query_sine_embed
      ]
      # Gets the cache for decoding.
      if cache is None:
        output_tensor, _ = self.decoder_layers[layer_idx](transformer_inputs, is_first=(layer_idx == 0))
      else:
        cache_layer_idx = str(layer_idx)
        output_tensor, cache[cache_layer_idx] = self.decoder_layers[layer_idx](
            transformer_inputs,
            cache=cache[cache_layer_idx],
            decode_loop_step=decode_loop_step,
            is_first=(layer_idx == 0))

      if self.iter_update:
        tmp = self.bbox_embed(output_tensor)  # bs, nq, 4
        tmp = tmp + inverse_sigmoid(reference_points)
        new_reference_points = tf.sigmoid(tmp)
        if layer_idx != self.num_layers - 1:
          ref_points.append(new_reference_points)
        reference_points = tf.stop_gradient(new_reference_points)

      if return_all_decoder_outputs:
        decoder_outputs.append(self.output_normalization(output_tensor))

    if return_all_decoder_outputs:
      if self.iter_update:
        return decoder_outputs, ref_points  # num_layers x (bs, num_queries, 4)
      else:
        return decoder_outputs, [reference_points for _ in range(self.num_layers)]
    else:
      return self.output_normalization(output_tensor)


class TransformerDecoderBlock(tf.keras.layers.Layer):
  """Single transformer layer for decoder.

  It has three sub-layers:
  (1) a multi-head self-attention mechanism.
  (2) a encoder-decoder attention.
  (3) a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               num_attention_heads,
               intermediate_size,
               intermediate_activation: str,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               kernel_initializer="he_uniform",
               bias_initializer=FanInBiasInitializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_epsilon=1e-12,
               intermediate_dropout=0.0,
               attention_initializer=None,
               is_first=False,
               **kwargs):
    """Initialize a Transformer decoder block.

    Args:
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate layer.
      intermediate_activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability for the post-attention and output
        dropout.
      attention_dropout_rate: Dropout probability for within the attention
        layer.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      attention_initializer: Initializer for kernels of attention layers. If set
        `None`, attention layers use kernel_initializer as initializer for
        kernel.
      **kwargs: key word arguemnts passed to tf.keras.layers.Layer.
    """
    super().__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    if intermediate_activation != "prelu":
      self.intermediate_activation = tf.keras.activations.get(
          intermediate_activation)
    else:
      self.intermediate_activation = "prelu"
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout
    self._is_first = is_first
    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = tf_utils.clone_initializer(
          self._kernel_initializer)
    self._cross_attention_cls = layers.attention.MultiHeadAttention

  def build(self, input_shape):
    target_tensor_shape = tf.TensorShape(input_shape[0])
    if len(target_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerLayer expects a three-dimensional input of "
                       "shape [batch, sequence, width].")
    hidden_size = target_tensor_shape[2]
    if hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self.num_attention_heads))
    self.attention_head_size = int(hidden_size) // self.num_attention_heads
    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    # Self attention.
    self.sa_qcontent_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="sa_qcontent_proj",
    )
    self.sa_qpos_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="sa_qpos_proj",
    )
    self.sa_kcontent_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="sa_kcontent_proj",
    )
    self.sa_kpos_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="sa_kpos_proj",
    )
    self.sa_v_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="sa_v_proj",
    )
    self.self_attention = layers.attention.CachedAttention(
        num_heads=self.num_attention_heads,
        key_dim=self.attention_head_size,
        dropout=self.attention_dropout_rate,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        name="self_attention",
        **common_kwargs)
    self.self_attention_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_rate)
    self.self_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype="float32"))
    # Encoder-decoder attention.
    self.ca_qcontent_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="ca_qcontent_proj",
    )
    if self._is_first:
      self.ca_qpos_proj = tf.keras.layers.Dense(
        units=hidden_size,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        activation=None,  # Linear activation (default)
        name="ca_qpos_proj",
      )
    else:
      self.ca_qpos_proj = None
    self.ca_kcontent_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="ca_kcontent_proj",
    )
    self.ca_kpos_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="ca_kpos_proj",
    )
    self.ca_v_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="ca_v_proj",
    )
    self.ca_qpos_sine_proj = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="ca_qpos_sine_proj",
    )
    self.encdec_attention = self._cross_attention_cls(
        num_heads=self.num_attention_heads,
        key_dim=self.attention_head_size,
        dropout=self.attention_dropout_rate,
        output_shape=hidden_size,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        name="attention/encdec",
        **common_kwargs)

    self.encdec_attention_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_rate)
    self.encdec_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="attention/encdec_output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype="float32"))

    # Feed-forward projection.
    self.intermediate_dense = tf.keras.layers.Dense(
      units=self.intermediate_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="intermediate",
    )
    if self.intermediate_activation != "prelu":
      self.intermediate_activation_layer = tf.keras.layers.Activation(
          self.intermediate_activation)
    else:
      self.intermediate_activation_layer = tf.keras.layers.PReLU(
          alpha_initializer=tf.keras.initializers.Constant(0.25))
    self._intermediate_dropout_layer = tf.keras.layers.Dropout(
        rate=self._intermediate_dropout)
    self.output_dense = tf.keras.layers.Dense(
      units=hidden_size,
      kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
      bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
      activation=None,  # Linear activation (default)
      name="output",
    )
    self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype="float32")
    super().build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads": self.num_attention_heads,
        "intermediate_size": self.intermediate_size,
        "intermediate_activation": tf_utils.serialize_activation(
            self.intermediate_activation, use_legacy_format=True
        ),
        "dropout_rate": self.dropout_rate,
        "attention_dropout_rate": self.attention_dropout_rate,
        "kernel_initializer": tf_utils.serialize_initializer(
            self._kernel_initializer, use_legacy_format=True
        ),
        "bias_initializer": tf_utils.serialize_initializer(
            self._bias_initializer, use_legacy_format=True
        ),
        "kernel_regularizer": tf_utils.serialize_regularizer(
            self._kernel_regularizer, use_legacy_format=True
        ),
        "bias_regularizer": tf_utils.serialize_regularizer(
            self._bias_regularizer, use_legacy_format=True
        ),
        "activity_regularizer": tf_utils.serialize_regularizer(
            self._activity_regularizer, use_legacy_format=True
        ),
        "kernel_constraint": tf_utils.serialize_constraint(
            self._kernel_constraint, use_legacy_format=True
        ),
        "bias_constraint": tf_utils.serialize_constraint(
            self._bias_constraint, use_legacy_format=True
        ),
        "use_bias": self._use_bias,
        "norm_epsilon": self._norm_epsilon,
        "intermediate_dropout": self._intermediate_dropout,
        "attention_initializer": tf_utils.serialize_initializer(
            self._attention_initializer, use_legacy_format=True
        ),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def common_layers_with_encoder(self):
    """Gets layer objects that can make a Transformer encoder block."""
    return [
        self.self_attention, self.self_attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_layer_norm
    ]

  def call(self, inputs, cache=None, decode_loop_step=None, is_first=False):
    # input_tensor is tgt, input_pos_embed is query_pos, memory_pos_embed is pos.
    input_tensor, memory, attention_mask, self_attention_mask, input_pos_embed, memory_pos_embed, query_sine_embed = inputs
    # shape: batch_size x num_queries x 256

    # ========== Begin of Self-Attention =============
    q_content = self.sa_qcontent_proj(input_tensor)  # target is the input of the first decoder layer. zero by default.
    q_pos = self.sa_qpos_proj(input_pos_embed)
    k_content = self.sa_kcontent_proj(input_tensor)
    k_pos = self.sa_kpos_proj(input_pos_embed)
    v = self.sa_v_proj(input_tensor)

    q = q_content + q_pos
    k = k_content + k_pos

    self_attention_output, cache = self.self_attention(
        query=q,
        key=k,
        value=v,
        attention_mask=self_attention_mask,
        cache=cache,
        decode_loop_step=decode_loop_step)

    self_attention_output = self.self_attention_layer_norm(
        input_tensor + self.self_attention_dropout(self_attention_output))
    # ========== End of Self-Attention =============

    # ========== Begin of Cross-Attention =============
    # Apply projections here
    # shape:  batch_size x num_queries x 256
    q_content = self.ca_qcontent_proj(self_attention_output)
    k_content = self.ca_kcontent_proj(memory)
    v = self.ca_v_proj(memory)

    # bs, num_queries, n_model = tf.shape(q_content)
    q_content_shape = tf.shape(q_content)
    bs = q_content_shape[0]
    num_queries = q_content_shape[1]
    n_model = q_content_shape[2]

    hw= tf.shape(k_content)[1]

    k_pos = self.ca_kpos_proj(memory_pos_embed)

    if is_first:
      q_pos = self.ca_qpos_proj(input_pos_embed)
      q = q_content + q_pos
      k = k_content + k_pos
    else:
      q = q_content
      k = k_content

    q = tf.reshape(q, (bs, num_queries, self.num_attention_heads, n_model // self.num_attention_heads))
    query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
    query_sine_embed = tf.reshape(query_sine_embed, (bs, num_queries, self.num_attention_heads, n_model // self.num_attention_heads))
    q = tf.reshape(tf.concat([q, query_sine_embed], axis=3), (bs, num_queries, n_model * 2))
    k = tf.reshape(k, (bs, hw, self.num_attention_heads, n_model // self.num_attention_heads))
    k_pos = tf.reshape(k_pos, (bs, hw, self.num_attention_heads, n_model // self.num_attention_heads))
    k = tf.reshape(tf.concat([k, k_pos], axis=3), (bs, hw, n_model * 2))

    cross_attn_inputs = dict(
        query=q,
        key=k,
        value=v,
        attention_mask=attention_mask)
    attention_output = self.encdec_attention(**cross_attn_inputs)
    attention_output = self.encdec_attention_dropout(attention_output)

    attention_output = self.encdec_attention_layer_norm(
        self_attention_output + attention_output)
    # ========== End of Cross-Attention =============

    intermediate_output = self.intermediate_dense(attention_output)
    intermediate_output = self.intermediate_activation_layer(
        intermediate_output)
    intermediate_output = self._intermediate_dropout_layer(intermediate_output)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = self.output_layer_norm(layer_output + attention_output)
    return layer_output, cache
