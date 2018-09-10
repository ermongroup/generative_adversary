"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MadryModel(object):
    def __init__(self, images):

        self.x_image = images

        # first convolutional layer
        W_conv1 = self._weight_variable([5, 5, 1, 32], name="Variable")
        b_conv1 = self._bias_variable([32], name="Variable_1")

        h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        # second convolutional layer
        W_conv2 = self._weight_variable([5, 5, 32, 64], name="Variable_2")
        b_conv2 = self._bias_variable([64], name="Variable_3")

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        # first fully connected layer
        W_fc1 = self._weight_variable([7 * 7 * 64, 1024], name="Variable_4")
        b_fc1 = self._bias_variable([1024], name="Variable_5")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # output layer
        W_fc2 = self._weight_variable([1024, 10], name="Variable_6")
        b_fc2 = self._bias_variable([10], name="Variable_7")

        self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.y_pred = tf.argmax(self.pre_softmax, 1)


    @staticmethod
    def _weight_variable(shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    @staticmethod
    def _bias_variable(shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=0.1))

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
