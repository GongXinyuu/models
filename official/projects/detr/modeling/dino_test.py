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

"""Tests for tensorflow_models.official.projects.detr.detr."""
import tensorflow as tf
from official.projects.detr.modeling import dino
from official.vision.modeling.backbones import resnet


class DinoTest(tf.test.TestCase):

  def test_forward(self):
    num_queries = 10
    hidden_size = 128
    num_classes = 10
    image_size = 640
    batch_size = 2
    backbone = resnet.ResNet(50, bn_trainable=False)
    backbone_endpoint_name = '5'
    model = dino.DINO(backbone, backbone_endpoint_name, num_queries,
                      hidden_size, num_classes)
    outs = model(tf.ones((batch_size, image_size, image_size, 3)))
    self.assertLen(outs, 6)  # intermediate decoded outputs.
    for out in outs:
      self.assertAllEqual(
          tf.shape(out['cls_outputs']), (batch_size, num_queries, num_classes))
      self.assertAllEqual(
          tf.shape(out['box_outputs']), (batch_size, num_queries, 4))

  def test_get_from_config_dino_transformer(self):
    config = {
        'num_encoder_layers': 1,
        'num_decoder_layers': 2,
        'dropout_rate': 0.5,
        "query_dim": 4,
        "keep_query_pos": False,
        "query_scale_type": 'cond_elewise',
        "modulate_hw_attn": True,
        "num_patterns": 0,
        "bbox_embed_diff_each_layer": False,
    }
    detr_model = dino.DINOTransformer.from_config(config)
    retrieved_config = detr_model.get_config()

    self.assertEqual(config, retrieved_config)

  def test_get_from_config_dino(self):
    config = {
        'backbone': resnet.ResNet(50, bn_trainable=False),
        'backbone_endpoint_name': '5',
        'num_queries': 2,
        'hidden_size': 4,
        'num_classes': 10,
        'num_encoder_layers': 4,
        'num_decoder_layers': 5,
        'dropout_rate': 0.5,
    }
    detr_model = dino.DINO.from_config(config)
    retrieved_config = detr_model.get_config()

    self.assertEqual(config, retrieved_config)


if __name__ == '__main__':
  tf.test.main()
