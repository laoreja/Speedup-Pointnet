"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

class PointCloudDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, tfrecord_path, mode, batch_size,
                  buffer_size=1000):
        """Create a new ImageDataGenerator.
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.
        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """        
        # create dataset
        data = tf.contrib.data.TFRecordDataset(tfrecord_path)
        
        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=4, output_buffer_size=100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=4, output_buffer_size=100*batch_size)
        else:
          raise ValueError("Invalid mode '%s'." % (mode))
            

        # number of samples in the dataset
        self.data_size = 9840 if mode == 'training' else 2468

        # shuffle the first `buffer_size` elements of the dataset
        if mode == 'training':
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _parse_function_train(self, serialized_example):
        """Input parser for samples of the training set."""
        points, label = self._parse_function_inference(serialized_example)
        points = self._rotate_point_cloud(points)
        points = self._jitter_point_cloud(points)
        return points, label

    def _parse_function_inference(self, serialized_example):
        """Input parser for samples of the validation/test set."""
#        print type(serialized_example)  <class 'tensorflow.python.framework.ops.Tensor'>
#        print serialized_example  Tensor("arg0:0", shape=(), dtype=string)
        feature_map = {
            'data': tf.FixedLenFeature([1024*3], dtype=tf.float32),
            'label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        }
        features = tf.parse_single_example(serialized_example, feature_map)
        label = tf.cast(features['label'], dtype=tf.int32)
        points = tf.reshape(features['data'], (-1, 3))
#        example = tf.train.Example()
#        example.ParseFromString(serialized_example)
#        points = np.array(example.features.feature['data'].float_list.value)        
#        label = np.array(example.features.feature['label'].int64_list.value)
        return points, label
        
    def _rotate_point_cloud(self, single_data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        rotation_angle = tf.random_uniform([], 0, 1) * 2 * np.pi
        rotation_matrix = tf.stack([tf.cos(rotation_angle), 0, tf.sin(rotation_angle),
                          0,1,0,
                          -tf.sin(rotation_angle),0, tf.cos(rotation_angle)])
        rotation_matrix = tf.reshape(rotation_matrix, (3,3))
        return tf.matmul(single_data, rotation_matrix)

    def _jitter_point_cloud(self, single_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = single_data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        single_data += jittered_data
        return single_data
