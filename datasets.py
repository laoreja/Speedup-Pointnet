# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Benchmark dataset utilities.
"""

from abc import abstractmethod
import os

import numpy as np
import tensorflow as tf
import preprocessing


MODELNET40_NUM_TRAIN = 9840
MODELNET40_NUM_VAL = 2468


def create_dataset(data_dir, data_name):
  """Create a Dataset instance based on data_dir and data_name."""
  supported_datasets = {
    'modelnet40cls': Modelnet40ClsData,
  }

  if data_name is None:
    for supported_name in supported_datasets:
      if supported_name in data_dir:
        data_name = supported_name
        break

  if data_name is None:
    raise ValueError('Could not identify name of dataset. '
                      'Please specify with --data_name option.')

  if data_name not in supported_datasets:
    raise ValueError('Unknown dataset. Must be one of %s', ', '.join(
        [key for key in sorted(supported_datasets.keys())]))

  return supported_datasets[data_name](data_dir)


class Dataset(object):
  """Abstract class for cnn benchmarks dataset."""

  def __init__(self, name, num_point_per_example data_dir=None,
                queue_runner_required=False, num_classes=40):
    self.name = name
    self.num_point_per_example = num_point_per_example
    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required
    self._num_classes = num_classes

  def reader(self):
    return tf.TFRecordReader()

  @property
  def num_classes(self):
    return self._num_classes

  @num_classes.setter
  def num_classes(self, val):
    self._num_classes = val

  @abstractmethod
  def num_examples_per_epoch(self, subset):
    pass

  def __str__(self):
    return self.name

  def get_image_preprocessor(self):
    return None

  def queue_runner_required(self):
    return self._queue_runner_required




class Modelnet40ClsData(Dataset):
  """Configuration for Modelnet40 classification dataset."""

  def __init__(self, data_dir=None):
    if data_dir is None:
      raise ValueError('Data directory not specified')
    super(Modelnet40ClsData, self).__init__('modelnet40Cls', num_point_per_example = 1024, data_dir=data_dir)

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return MODELNET40_NUM_TRAIN
    elif subset == 'validation':
      return MODELNET40_NUM_VAL
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_image_preprocessor(self):
    return preprocessing.RecordInputImagePreprocessor

