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

"""Tests for dino."""

# pylint: disable=unused-import
from absl.testing import parameterized
import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.detr.configs import dino as exp_cfg
from official.projects.detr.dataloaders import coco


class DinoTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('dino_coco',))
  def test_dino_configs_tfds(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task, exp_cfg.DinoTask)
    self.assertIsInstance(config.task.train_data, coco.COCODataConfig)
    config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      config.validate()

  @parameterized.parameters(('dino_coco_tfrecord'), ('dino_coco_tfds'))
  def test_dino_configs(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task, exp_cfg.DinoTask)
    self.assertIsInstance(config.task.train_data, cfg.DataConfig)
    config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      config.validate()


if __name__ == '__main__':
  tf.test.main()
