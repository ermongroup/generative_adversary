from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import variable_scope
import tensorflow as tf


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dataset='cifar',
           scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                outputs_collections=end_points_collection):
            def ConvBatchRelu(layer_input, n_output_plane, name):
                with variable_scope.variable_scope(name):
                    output = layers.conv2d(layer_input, n_output_plane, [3, 3], scope='conv')
                    output = layers.batch_norm(output, center=True, scale=True, activation_fn=tf.nn.relu,
                                               is_training=is_training)
                return output

            filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]
            if dataset == 'mnist':
                filters = [_ // 4 for _ in filters]
            elif dataset not in ('cifar', 'svhn'):
                raise NotImplementedError("Dataset {} is not supported!".format(dataset))

            net = ConvBatchRelu(inputs, filters[0], 'conv1_1')
            net = ConvBatchRelu(net, filters[1], 'conv1_2')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
            net = ConvBatchRelu(net, filters[2], 'conv2_1')
            net = ConvBatchRelu(net, filters[3], 'conv2_2')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
            net = ConvBatchRelu(net, filters[4], 'conv3_1')
            net = ConvBatchRelu(net, filters[5], 'conv3_2')
            net = ConvBatchRelu(net, filters[6], 'conv3_3')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
            net = ConvBatchRelu(net, filters[7], 'conv4_1')
            net = ConvBatchRelu(net, filters[8], 'conv4_2')
            net = ConvBatchRelu(net, filters[9], 'conv4_3')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
            net = ConvBatchRelu(net, filters[10], 'conv5_1')
            net = ConvBatchRelu(net, filters[11], 'conv5_2')
            net = ConvBatchRelu(net, filters[12], 'conv5_3')
            if dataset == 'cifar':
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = layers.flatten(net, scope='flatten6')
            net = layers_lib.dropout(net, 0.5, is_training=is_training, scope='dropout6')
            net = layers.relu(net, filters[13])
            net = layers_lib.dropout(net, 0.5, is_training=is_training, scope='dropout6')
            net = layers.linear(net, num_classes)
            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)
            end_points[sc.name + '/fc8'] = net
            return net, end_points

vgg_16.default_image_size = 32
