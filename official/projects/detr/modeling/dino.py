# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Implements End-to-End Object Detection with Transformers.

Model paper: https://arxiv.org/abs/2005.12872
This module does not support Keras de/serialization. Please use
tf.train.Checkpoint for object based saving and loading and tf.saved_model.save
for graph serializaiton.
"""
import math
from typing import Any, List, Dict, Union

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.detr.modeling import transformer_dino
from official.projects.detr.modeling import transformer
from official.vision.ops import box_ops


def position_embedding_sine(attention_mask,
                            num_pos_features=256,
                            temperature=10000.,
                            normalize=True,
                            scale=2 * math.pi):
  """Sine-based positional embeddings for 2D images.

  Args:
    attention_mask: a `bool` Tensor specifying the size of the input image to
      the Transformer and which elements are padded, of size [batch_size,
      height, width]
    num_pos_features: a `int` specifying the number of positional features,
      should be equal to the hidden size of the Transformer network
    temperature: a `float` specifying the temperature of the positional
      embedding. Any type that is converted to a `float` can also be accepted.
    normalize: a `bool` determining whether the positional embeddings should be
      normalized between [0, scale] before application of the sine and cos
      functions.
    scale: a `float` if normalize is True specifying the scale embeddings before
      application of the embedding function.

  Returns:
    embeddings: a `float` tensor of the same shape as input_tensor specifying
      the positional embeddings based on sine features.
  """
  if num_pos_features % 2 != 0:
    raise ValueError(
        "Number of embedding features (num_pos_features) must be even when "
        "column and row embeddings are concatenated.")
  num_pos_features = num_pos_features // 2

  # Produce row and column embeddings based on total size of the image
  # <tf.float>[batch_size, height, width]
  attention_mask = tf.cast(attention_mask, tf.float32)
  row_embedding = tf.cumsum(attention_mask, 1)
  col_embedding = tf.cumsum(attention_mask, 2)

  if normalize:
    eps = 1e-6
    row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
    col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

  dim_t = tf.range(num_pos_features, dtype=row_embedding.dtype)
  dim_t = tf.pow(temperature, 2 * (dim_t // 2) / num_pos_features)

  # Creates positional embeddings for each row and column position
  # <tf.float>[batch_size, height, width, num_pos_features]
  pos_row = tf.expand_dims(row_embedding, -1) / dim_t
  pos_col = tf.expand_dims(col_embedding, -1) / dim_t
  pos_row = tf.stack(
      [tf.sin(pos_row[:, :, :, 0::2]),
       tf.cos(pos_row[:, :, :, 1::2])], axis=4)
  pos_col = tf.stack(
      [tf.sin(pos_col[:, :, :, 0::2]),
       tf.cos(pos_col[:, :, :, 1::2])], axis=4)

  # final_shape = pos_row.shape.as_list()[:3] + [-1]
  final_shape = tf_utils.get_shape_list(pos_row)[:3] + [-1]
  pos_row = tf.reshape(pos_row, final_shape)
  pos_col = tf.reshape(pos_col, final_shape)
  output = tf.concat([pos_row, pos_col], -1)

  embeddings = tf.cast(output, tf.float32)
  return embeddings


def refpoint_initializer(shape, dtype=None):
  weights = tf.random.normal(shape, mean=0., stddev=1., dtype=dtype)

  # Uniformly initialize the first two columns
  weights[:, :2] = tf.random.uniform((shape[0], 2), minval=0, maxval=1, dtype=dtype)

  # Apply inverse_sigmoid function
  weights[:, :2] = transformer_dino.inverse_sigmoid(weights[:, :2])

  # Set the first two columns to not require gradient updates
  weights[:, :2] = tf.stop_gradient(weights[:, :2])

  return weights


def postprocess(outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Performs post-processing on model output.

  Args:
    outputs: The raw model output.

  Returns:
    Postprocessed model output.
  """
  predictions = {
      "detection_boxes":  # Box coordinates are relative values here.
          box_ops.cycxhw_to_yxyx(outputs["box_outputs"]),
      "detection_scores":
          tf.math.reduce_max(
              tf.nn.softmax(outputs["cls_outputs"])[:, :, 1:], axis=-1),
      "detection_classes":
          tf.math.argmax(outputs["cls_outputs"][:, :, 1:], axis=-1) + 1,
      # Fix this. It's not being used at the moment.
      "num_detections":
          tf.reduce_sum(
              tf.cast(
                  tf.math.greater(
                      tf.math.reduce_max(outputs["cls_outputs"], axis=-1), 0),
                  tf.int32),
              axis=-1)
  }
  return predictions


class DINO(tf.keras.Model):
  """DINO model with Keras.

  DINO consists of backbone, query embedding, DETRTransformer,
  class and box heads.
  """

  def __init__(self,
               backbone,
               backbone_endpoint_name,
               num_queries,
               hidden_size,
               num_classes,
               num_encoder_layers=6,
               num_decoder_layers=6,
               dropout_rate=0.1,
               query_dim=4,
               keep_query_pos=False,
               query_scale_type='cond_elewise',
               num_patterns=0,
               modulate_hw_attn=True,
               bbox_embed_diff_each_layer=False,
               random_refpoints_xy=False,
               **kwargs):
    super().__init__(**kwargs)
    assert query_dim in [2, 4]

    self._num_queries = num_queries
    self._hidden_size = hidden_size
    self._num_classes = num_classes
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._dropout_rate = dropout_rate
    self._query_dim = query_dim
    self._keep_query_pos = keep_query_pos
    self._query_scale_type = query_scale_type
    self._num_patterns = num_patterns
    self._modulate_hw_attn = modulate_hw_attn
    self._bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
    self._random_refpoints_xy = random_refpoints_xy
    if hidden_size % 2 != 0:
      raise ValueError("hidden_size must be a multiple of 2.")
    self._backbone = backbone
    self._backbone_endpoint_name = backbone_endpoint_name

  def build(self, input_shape=None):
    self._input_proj = tf.keras.layers.Conv2D(
        self._hidden_size, 1, name="dino/conv2d")
    self._build_detection_decoder()
    super().build(input_shape)

  def _build_detection_decoder(self):
    """Builds detection decoder."""
    self._transformer = DINOTransformer(
        num_encoder_layers=self._num_encoder_layers,
        num_decoder_layers=self._num_decoder_layers,
        dropout_rate=self._dropout_rate,
        query_dim=self._query_dim,
        keep_query_pos=self._keep_query_pos,
        query_scale_type=self._query_scale_type,
        num_patterns=self._num_patterns,
        modulate_hw_attn=self._modulate_hw_attn,
        bbox_embed_diff_each_layer=self._bbox_embed_diff_each_layer)

    self.refpoint_embed = self.add_weight(
      "dino/refpoint_embeddings",
      shape=[self._num_queries, self._query_dim],
      initializer=refpoint_initializer if self._random_refpoints_xy else None,
      dtype=tf.float32
    )

    sqrt_k = math.sqrt(1.0 / self._hidden_size)
    self._class_embed = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomUniform(-sqrt_k, sqrt_k),
        name="dino/cls_dense")

    self._sigmoid = tf.keras.layers.Activation("sigmoid")

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  @property
  def _bbox_embed(self) -> Union[tf.keras.Model, List[tf.keras.Model]]:
    return self._transformer._decoder.bbox_embed

  def get_config(self):
    return {
        "backbone": self._backbone,
        "backbone_endpoint_name": self._backbone_endpoint_name,
        "num_queries": self._num_queries,
        "hidden_size": self._hidden_size,
        "num_classes": self._num_classes,
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def _generate_image_mask(self, inputs: tf.Tensor,
                           target_shape: tf.Tensor) -> tf.Tensor:
    """Generates image mask from input image."""
    mask = tf.expand_dims(
        tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0), inputs.dtype),
        axis=-1)
    mask = tf.image.resize(
        mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mask

  def call(self, inputs: tf.Tensor, training: bool = None) -> List[Any]:
    batch_size = tf.shape(inputs)[0]
    features = self._backbone(inputs)[self._backbone_endpoint_name]
    shape = tf.shape(features)
    mask = self._generate_image_mask(inputs, shape[1: 3])
    embedweight = tf.tile(tf.expand_dims(self.refpoint_embed, axis=0), (batch_size, 1, 1))

    pos_embed = position_embedding_sine(
        mask[:, :, :, 0], num_pos_features=self._hidden_size)
    pos_embed = tf.reshape(pos_embed, [batch_size, -1, self._hidden_size])

    features = tf.reshape(
        self._input_proj(features), [batch_size, -1, self._hidden_size])
    mask = tf.reshape(mask, [batch_size, -1])

    decoded_list, reference_list = self._transformer({
        "inputs":
            features,
        "targets":
            embedweight,
        "pos_embed": pos_embed,
        "mask": mask,
    })

    out_list = []
    for layer_idx, (decoded, reference) in enumerate(zip(decoded_list, reference_list)):
      decoded = tf.stack(decoded)  # bs, num_queries, hidden_size, no effect
      output_class = self._class_embed(decoded)  # bs, num_queries, num_classes

      if not self._bbox_embed_diff_each_layer:
        reference_before_sigmoid = transformer_dino.inverse_sigmoid(reference)
        tmp = self._bbox_embed(decoded)

        # tmp[..., :self._query_dim] += reference_before_sigmoid, original implementation
        # TODO: check if this is correct
        output_coord = self._sigmoid(tmp + reference_before_sigmoid)
      else:
        reference_before_sigmoid = transformer_dino.inverse_sigmoid(reference)
        tmp = self._bbox_embed[layer_idx](decoded)
        # tmp[..., :self._query_dim] += reference_before_sigmoid
        # TODO: check if this is correct
        output_coord = self._sigmoid(tmp + reference_before_sigmoid)

      out = {"cls_outputs": output_class, "box_outputs": output_coord}
      if not training:
        out.update(postprocess(out))
      out_list.append(out)

    return out_list


class DINOTransformer(tf.keras.layers.Layer):
  """Encoder and Decoder of DINO."""

  def __init__(self, num_encoder_layers=6, num_decoder_layers=6,
               dropout_rate=0.1, query_dim=4, keep_query_pos=False,
               query_scale_type='cond_elewise', num_patterns=0,
               modulate_hw_attn=True, bbox_embed_diff_each_layer=False, **kwargs):
    super().__init__(**kwargs)
    assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

    self._dropout_rate = dropout_rate
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._query_dim = query_dim
    self._keep_query_pos = keep_query_pos
    self._query_scale_type = query_scale_type
    self._modulate_hw_attn = modulate_hw_attn
    self._num_patterns = num_patterns
    self._bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

  def build(self, input_shape=None):
    pos_embed_tensor_shape = tf.TensorShape(input_shape['pos_embed'])
    if len(pos_embed_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerLayer expects a three-dimensional input of "
                       "shape [batch, sequence, width].")
    hidden_size = pos_embed_tensor_shape[2]
    self._hidden_size = hidden_size
    if self._num_encoder_layers > 0:
      self._encoder = transformer.TransformerEncoder(
          attention_dropout_rate=self._dropout_rate,
          dropout_rate=self._dropout_rate,
          intermediate_dropout=self._dropout_rate,
          norm_first=False,
          num_layers=self._num_encoder_layers)
    else:
      self._encoder = None

    self._decoder = transformer_dino.TransformerDecoder(
        attention_dropout_rate=self._dropout_rate,
        dropout_rate=self._dropout_rate,
        intermediate_dropout=self._dropout_rate,
        num_layers=self._num_decoder_layers,
        query_dim=self._query_dim,
        keep_query_pos=self._keep_query_pos,
        query_scale_type=self._query_scale_type,
        modulate_hw_attn=self._modulate_hw_attn,
        bbox_embed_diff_each_layer=self._bbox_embed_diff_each_layer)

    if self._num_patterns > 0:
      # self._patterns = tf.keras.layers.Embedding(self._num_patterns, hidden_size)
      self._patterns = self.add_weight(
        "patterns",
        shape=[self._num_patterns, hidden_size],
        dtype=tf.float32
      )

    super().build(input_shape)

  def get_config(self):
    return {
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
        "query_dim": self._query_dim,
        "keep_query_pos": self._keep_query_pos,
        "query_scale_type": self._query_scale_type,
        "modulate_hw_attn": self._modulate_hw_attn,
        "num_patterns": self._num_patterns,
        "bbox_embed_diff_each_layer": self._bbox_embed_diff_each_layer,
    }

  def call(self, inputs):
    sources = inputs["inputs"]
    refpoint_embed = inputs["targets"]  # query embeddings, shape: (bs, num_queries, query_dim)
    pos_embed = inputs["pos_embed"]
    mask = inputs["mask"]
    input_shape = tf_utils.get_shape_list(sources)
    source_attention_mask = tf.tile(
        tf.expand_dims(mask, axis=1), [1, input_shape[1], 1])
    if self._encoder is not None:
      memory = self._encoder(
          sources, attention_mask=source_attention_mask, pos_embed=pos_embed)
    else:
      memory = sources

    target_shape = tf_utils.get_shape_list(refpoint_embed)
    cross_attention_mask = tf.tile(
        tf.expand_dims(mask, axis=1), [1, target_shape[1], 1])
    target_shape = tf.shape(refpoint_embed)

    bs = target_shape[0]
    num_queries = target_shape[1]
    if self._num_patterns == 0:
      tgt = tf.zeros((bs, num_queries, self._hidden_size))  # (bs, num_queries, hidden_size)
    else:
      # Get the embeddings and reshape
      tgt = self._patterns[:, tf.newaxis, tf.newaxis, :]
      tgt = tf.tile(tgt, [1, num_queries, bs, 1])  # (num_patterns, num_queries, bs, hidden_size)
      tgt = tf.reshape(tgt, (bs, -1, self._hidden_size))  # (bs, num_patterns * num_queries, hidden_size)
      refpoint_embed = tf.tile(refpoint_embed, (1, self._num_patterns, 1))

    decoded = self._decoder(
        tgt,
        memory,
        # TODO(b/199545430): self_attention_mask could be set to None when this
        # bug is resolved. Passing ones for now.
        self_attention_mask=tf.ones(
            (target_shape[0], target_shape[1], target_shape[1])),
        cross_attention_mask=cross_attention_mask,
        return_all_decoder_outputs=True,
        refpoints_unsigmoid=refpoint_embed,
        memory_pos_embed=pos_embed)
    return decoded
