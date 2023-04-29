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

"""DETR detection task definition."""
from typing import Optional

from absl import logging
import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.projects.detr.configs import dino as dino_cfg
from official.projects.detr.dataloaders import coco
from official.projects.detr.dataloaders import detr_input
from official.projects.detr.modeling import dino
from official.projects.detr.ops import matchers
from official.projects.detr.tasks import detection
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import tf_example_decoder
from official.vision.dataloaders import tfds_factory
from official.vision.dataloaders import tf_example_label_map_decoder
from official.vision.evaluation import coco_evaluator
from official.vision.modeling import backbones
from official.vision.ops import box_ops


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

