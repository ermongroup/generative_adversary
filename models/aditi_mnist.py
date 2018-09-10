"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class AditiMNIST(object):
    def __init__(self, images):
        self.x_image = images
        x = tf.reshape(self.x_image, shape=(-1, 784))
        W1 = tf.get_variable('W1', shape=(784, 500), dtype=tf.float32, initializer=tf.random_normal_initializer)
        B1 = tf.get_variable('B1', shape=(500,), dtype=tf.float32, initializer=tf.random_normal_initializer)
        W2 = tf.get_variable('W2', shape=(500, 10), dtype=tf.float32, initializer=tf.random_normal_initializer)
        B2 = tf.get_variable('B2', shape=(10,), dtype=tf.float32, initializer=tf.random_normal_initializer)

        y = x @ W1 + B1
        y = tf.nn.relu(y)
        self.logits = y @ W2 + B2