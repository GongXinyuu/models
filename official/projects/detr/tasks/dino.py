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

"""DINO detection task definition."""
import tensorflow as tf

from official.core import task_factory
from official.projects.detr.configs import dino as dino_cfg
from official.projects.detr.modeling import dino
from official.projects.detr.ops import matchers
from official.projects.detr.tasks import detection
from official.vision.modeling import backbones
from official.vision.ops import box_ops


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
  softmax_prob = tf.nn.softmax(logits, axis=-1)
  onehot_labels = tf.one_hot(labels, depth=tf.shape(softmax_prob)[-1])
  ce_loss = -tf.reduce_sum(onehot_labels * tf.math.log(softmax_prob + 1e-8), axis=-1)
  ce_loss = alpha * tf.pow(1 - softmax_prob, gamma) * tf.expand_dims(ce_loss, -1)
  return tf.reduce_mean(ce_loss)


@task_factory.register_task_cls(dino_cfg.DinoTask)
class DINOTask(detection.DetectionTask):
  """A single-replica view of training procedure.

  DETR task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Build DINO model."""

    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            self._task_config.model.input_size)

    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=self._task_config.model.backbone,
        norm_activation_config=self._task_config.model.norm_activation)

    model = dino.DINO(backbone,
                      self._task_config.model.backbone_endpoint_name,
                      self._task_config.model.num_queries,
                      self._task_config.model.hidden_size,
                      self._task_config.model.num_classes,
                      self._task_config.model.num_encoder_layers,
                      self._task_config.model.num_decoder_layers,
                      self._task_config.model.dropout_rate,
                      query_dim=self._task_config.model.query_dim,
                      keep_query_pos=self._task_config.model.keep_query_pos,
                      query_scale_type=self._task_config.model.query_scale_type,
                      num_patterns=self._task_config.model.num_patterns,
                      modulate_hw_attn=self._task_config.model.modulate_hw_attn,
                      bbox_embed_diff_each_layer=self._task_config.model.bbox_embed_diff_each_layer,
                      random_refpoints_xy=self._task_config.model.random_refpoints_xy)
    return model

  def build_losses(self, outputs, labels, aux_losses=None):
    """Builds DINO losses."""
    cls_outputs = outputs['cls_outputs']  # bs, num_queries, num_classes
    box_outputs = outputs['box_outputs']
    cls_targets = labels['classes']
    box_targets = labels['boxes']

    num_queries = tf.cast(tf.shape(cls_outputs)[1], tf.float32)

    cost = self._compute_cost(
        cls_outputs, box_outputs, cls_targets, box_targets)

    # Hungarian matching.
    _, indices = matchers.hungarian_matching(cost)
    indices = tf.stop_gradient(indices)

    target_index = tf.math.argmax(indices, axis=1)
    cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
    box_assigned = tf.gather(box_outputs, target_index, batch_dims=1, axis=1)

    background = tf.equal(cls_targets, 0)
    num_boxes = tf.reduce_sum(
        tf.cast(tf.logical_not(background), tf.float32), axis=-1)

    # Box loss is only calculated on non-background class.
    l_1 = tf.reduce_sum(tf.abs(box_assigned - box_targets), axis=-1)
    box_loss = self._task_config.losses.lambda_box * tf.where(
        background, tf.zeros_like(l_1), l_1)

    # Giou loss is only calculated on non-background class.
    giou = tf.linalg.diag_part(1.0 - box_ops.bbox_generalized_overlap(
        box_ops.cycxhw_to_yxyx(box_assigned),
        box_ops.cycxhw_to_yxyx(box_targets)
        ))
    giou_loss = self._task_config.losses.lambda_giou * tf.where(
        background, tf.zeros_like(giou), giou)

    # Consider doing all reduce once in train_step to speed up.
    num_boxes_per_replica = tf.reduce_sum(num_boxes)
    replica_context = tf.distribute.get_replica_context()
    num_boxes_sum = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM,
        num_boxes_per_replica)

    # focal loss for class imbalance.
    cls_loss = tf.math.divide_no_nan(
      num_queries * self._task_config.losses.lambda_cls * focal_loss(
      cls_assigned, cls_targets, alpha=self._task_config.losses.focal_alpha, gamma=self._task_config.losses.focal_gamma),
      num_boxes_sum)
    box_loss = tf.math.divide_no_nan(
        tf.reduce_sum(box_loss), num_boxes_sum)
    giou_loss = tf.math.divide_no_nan(
        tf.reduce_sum(giou_loss), num_boxes_sum)

    aux_losses = tf.add_n(aux_losses) if aux_losses else 0.0

    total_loss = cls_loss + box_loss + giou_loss + aux_losses
    return total_loss, cls_loss, box_loss, giou_loss