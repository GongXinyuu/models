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
from typing import List, Dict, Tuple, Union, Optional  # noqa
import tensorflow as tf

from official.core import task_factory
from official.projects.detr.configs import dino as dino_cfg
from official.projects.detr.modeling import dino
from official.projects.detr.ops import matchers
from official.projects.detr.tasks import detection
from official.vision.modeling import backbones
from official.vision.ops import box_ops
from official.vision.losses import focal_loss


@task_factory.register_task_cls(dino_cfg.DinoTask)
class DINOTask(detection.DetectionTask):
  """A single-replica view of training procedure.

  DETR task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """
  def __init__(self,
               params,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None):
    """Task initialization.

    Args:
      params: the task configuration instance, which can be any of dataclass,
        ConfigDict, namedtuple, etc.
      logging_dir: a string pointing to where the model, summaries etc. will be
        saved. You can also write additional stuff in this directory.
      name: the task name.
    """
    super().__init__(params=params, logging_dir=logging_dir, name=name)

    self._cls_loss_fn = focal_loss.FocalLoss(
      alpha=self._task_config.losses.focal_alpha,
      gamma=self._task_config.losses.focal_gamma,
      reduction=tf.keras.losses.Reduction.NONE,
      name='cls_loss')

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
                      num_patterns=self._task_config.model.num_patterns,
                      modulate_hw_attn=self._task_config.model.modulate_hw_attn,
                      bbox_embed_diff_each_layer=self._task_config.model.bbox_embed_diff_each_layer,
                      random_refpoints_xy=self._task_config.model.random_refpoints_xy)
    return model

  def _compute_cost(self, cls_outputs, box_outputs, cls_targets, box_targets):
    # Approximate classification cost with 1 - prob[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    # background: 0
    out_prob = tf.sigmoid(cls_outputs)
    neg_cost_class = (1 - self._task_config.losses.focal_alpha) * (out_prob ** self._task_config.losses.focal_gamma) \
                     * (-tf.math.log(1 - out_prob + 1e-8))
    pos_cost_class = self._task_config.losses.focal_alpha * ((1 - out_prob) ** self._task_config.losses.focal_gamma) \
                     * (-tf.math.log(out_prob + 1e-8))
    cls_cost = self._task_config.losses.lambda_cls_cost * \
               (tf.gather(pos_cost_class, cls_targets, batch_dims=1, axis=-1) - tf.gather(neg_cost_class, cls_targets, batch_dims=1, axis=-1))

    # Compute the L1 cost between boxes,
    paired_differences = self._task_config.losses.lambda_box_cost * tf.abs(
        tf.expand_dims(box_outputs, 2) - tf.expand_dims(box_targets, 1))
    box_cost = tf.reduce_sum(paired_differences, axis=-1)

    # Compute the giou cost betwen boxes
    giou_cost = self._task_config.losses.lambda_giou_cost * -box_ops.bbox_generalized_overlap(
        box_ops.cycxhw_to_yxyx(box_outputs),
        box_ops.cycxhw_to_yxyx(box_targets))

    total_cost = cls_cost + box_cost + giou_cost

    max_cost = (
        self._task_config.losses.lambda_cls_cost * 0.0 +
        self._task_config.losses.lambda_box_cost * 4. +
        self._task_config.losses.lambda_giou_cost * 0.0)

    # Set pads to large constant
    valid = tf.expand_dims(
        tf.cast(tf.not_equal(cls_targets, 0), dtype=total_cost.dtype), axis=1)
    total_cost = (1 - valid) * max_cost + valid * total_cost

    # Set inf of nan to large constant
    total_cost = tf.where(
        tf.logical_or(tf.math.is_nan(total_cost), tf.math.is_inf(total_cost)),
        max_cost * tf.ones_like(total_cost, dtype=total_cost.dtype),
        total_cost)

    return total_cost

  def build_losses(self, outputs, labels, aux_losses=None, lambda_cls=None, lambda_box=None, lambda_giou=None):
    """Builds DINO losses."""
    lambda_cls = self._task_config.losses.lambda_cls if lambda_cls is None else lambda_cls
    lambda_box = self._task_config.losses.lambda_box if lambda_box is None else lambda_box
    lambda_giou = self._task_config.losses.lambda_giou if lambda_giou is None else lambda_giou

    cls_outputs = outputs['cls_outputs']  # bs, num_queries, num_classes
    box_outputs = outputs['box_outputs']
    cls_targets = labels['classes']
    box_targets = labels['boxes']

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

    # focal loss for class imbalance.
    cls_targets_one_hot = tf.one_hot(cls_targets, self._task_config.model.num_classes)
    # cls_loss shape: [batch_size, num_queries, num_classes]
    cls_loss = lambda_cls * tf.reduce_sum(self._cls_loss_fn(cls_targets_one_hot, cls_assigned))

    # Box loss is only calculated on non-background class.
    l_1 = tf.reduce_sum(tf.abs(box_assigned - box_targets), axis=-1)
    box_loss = lambda_box * tf.where(background, tf.zeros_like(l_1), l_1)

    # Giou loss is only calculated on non-background class.
    giou = tf.linalg.diag_part(1.0 - box_ops.bbox_generalized_overlap(
        box_ops.cycxhw_to_yxyx(box_assigned),
        box_ops.cycxhw_to_yxyx(box_targets)
        ))
    giou_loss = lambda_giou * tf.where(background, tf.zeros_like(giou), giou)

    # Consider doing all reduce once in train_step to speed up.
    num_boxes_per_replica = tf.reduce_sum(num_boxes)
    replica_context = tf.distribute.get_replica_context()
    num_boxes_sum = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM,
        num_boxes_per_replica)

    cls_loss = tf.math.divide_no_nan(
      tf.reduce_sum(cls_loss), num_boxes_sum)
    box_loss = tf.math.divide_no_nan(
        tf.reduce_sum(box_loss), num_boxes_sum)
    giou_loss = tf.math.divide_no_nan(
        tf.reduce_sum(giou_loss), num_boxes_sum)

    aux_losses = tf.add_n(aux_losses) if aux_losses else 0.0

    total_loss = cls_loss + box_loss + giou_loss + aux_losses
    return total_loss, cls_loss, box_loss, giou_loss

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    with tf.GradientTape() as tape:
      outputs, interm_out = model(features, training=True)

      loss = 0.0
      cls_loss = 0.0
      box_loss = 0.0
      giou_loss = 0.0

      # final loss +  aux loss
      for output in outputs:
        # Computes per-replica loss.
        layer_loss, layer_cls_loss, layer_box_loss, layer_giou_loss = self.build_losses(
            outputs=output, labels=labels, aux_losses=model.losses)
        loss += layer_loss
        cls_loss += layer_cls_loss
        box_loss += layer_box_loss
        giou_loss += layer_giou_loss

      # compute intermediate loss
      layer_loss, layer_cls_loss, layer_box_loss, layer_giou_loss = self.build_losses(
        outputs=interm_out, labels=labels, aux_losses=model.losses)
      loss += layer_loss
      cls_loss += layer_cls_loss
      box_loss += layer_box_loss
      giou_loss += layer_giou_loss

      # Consider moving scaling logic from build_losses to here.
      scaled_loss = loss
      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient when LossScaleOptimizer is used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    # Multiply for logging.
    # Since we expect the gradient replica sum to happen in the optimizer,
    # the loss is scaled with global num_boxes and weights.
    # To have it more interpretable/comparable we scale it back when logging.
    num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
    loss *= num_replicas_in_sync
    cls_loss *= num_replicas_in_sync
    box_loss *= num_replicas_in_sync
    giou_loss *= num_replicas_in_sync

    # Trainer class handles loss metric for you.
    logs = {self.loss: loss}

    all_losses = {
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'giou_loss': giou_loss,
    }

    # Metric results will be added to logs for you.
    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
    return logs