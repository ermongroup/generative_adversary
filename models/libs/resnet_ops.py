import numpy as np
import tensorflow as tf
from models.libs.sn import spectral_normed_weight
from models.libs.ops import gan_cond_batch_norm
from functools import partial

def conv2d(input_, output_dim, filter_size, stddev=None,
           name="conv2d", spectral_normed=False, update_collection=None, with_w=False, he_init=True, padding="SAME"):
    # Glorot intialization
    # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    k_h = filter_size
    k_w = filter_size
    d_h = 1
    d_w = 1
    fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
    fan_out = k_h * k_w * output_dim / (d_h * d_w)
    if stddev is None:
        if he_init:
            stddev = np.sqrt(4. / (fan_in + fan_out)) # He initialization
        else:
            stddev = np.sqrt(2. / (fan_in + fan_out)) # Glorot initialization

    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_uniform_initializer(
                                minval=-stddev * np.sqrt(3),
                                maxval=stddev * np.sqrt(3)
                            ))
        if spectral_normed:
            conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                                strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases
        else:
            return conv

def ConvMeanPool(input, output_dim, filter_size, name, spectral_normed=False, update_collection=None, he_init=True):
    output = conv2d(input, output_dim, filter_size, spectral_normed=spectral_normed,
                    update_collection=update_collection, name=name, he_init=he_init)
    output = tf.add_n([output[:, ::2, ::2, :],
                       output[:, 1::2, ::2, :],
                       output[:, ::2, 1::2, :],
                       output[:, 1::2, 1::2, :]]) / 4.
    return output

def MeanPoolConv(input, output_dim, filter_size, name, spectral_normed=False, update_collection=None, he_init=True):
    output = input
    output = tf.add_n([output[:, ::2, ::2, :],
                       output[:, 1::2, ::2, :],
                       output[:, ::2, 1::2, :],
                       output[:, 1::2, 1::2, :]]) / 4.
    return conv2d(output, output_dim, filter_size, spectral_normed=spectral_normed,
                  update_collection=update_collection, name=name, he_init=he_init)

def UpsampleConv(input, output_dim, filter_size, name, spectral_normed=False, update_collection=None, he_init=True):
    output = input
    output = tf.concat([output, output, output, output], axis=3)
    output = tf.depth_to_space(output, 2)
    return conv2d(output, output_dim, filter_size, spectral_normed=spectral_normed,
                  update_collection=update_collection, name=name, he_init=he_init)

def ResidualBlock_celebA(input, labels, output_dim, filter_size, resample, name, spectral_normed=False, update_collection=None, n_labels=10):
    input_dim = input.get_shape().as_list()[-1]
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv1 = partial(conv2d, output_dim=input_dim)
        conv2 = partial(ConvMeanPool, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv1 = partial(UpsampleConv, output_dim=output_dim)
        conv2 = partial(conv2d, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = conv2d
        conv1 = partial(conv2d, output_dim=input_dim)
        conv2 = partial(conv2d, output_dim=output_dim)

    if output_dim == input_dim and resample is None:
        shortcut = input
    else:
        shortcut = conv_shortcut(input, output_dim=output_dim, filter_size=1, spectral_normed=spectral_normed,
                                 update_collection=update_collection, he_init=False, name=name+'_shortcut')

    output = input
    if labels is not None:
        output = gan_cond_batch_norm(output, labels, n_labels=n_labels, name=name+'_bn1')

    output = tf.nn.relu(output)
    output = conv1(output, filter_size=filter_size, spectral_normed=spectral_normed,
                   update_collection=update_collection, name=name+'_conv1')

    if labels is not None:
        output = gan_cond_batch_norm(output, labels, n_labels=n_labels, name=name+'_bn2')

    output = tf.nn.relu(output)
    output = conv2(output, filter_size=filter_size, spectral_normed=spectral_normed,
                   update_collection=update_collection, name=name+'_conv2')

    return output + shortcut

def ResidualBlock(input, labels, output_dim, filter_size, resample, name, spectral_normed=False, update_collection=None, n_labels=10):
    input_dim = input.get_shape().as_list()[-1]
    if resample == 'down':
        conv1 = partial(conv2d, output_dim=input_dim)
        conv2 = partial(ConvMeanPool, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv1 = partial(UpsampleConv, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv2 = partial(conv2d, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = conv2d
        conv1 = partial(conv2d, output_dim=output_dim)
        conv2 = partial(conv2d, output_dim=output_dim)

    if output_dim == input_dim and resample is None:
        shortcut = input
    else:
        shortcut = conv_shortcut(input, output_dim=output_dim, filter_size=1, spectral_normed=spectral_normed,
                                 update_collection=update_collection, he_init=False, name=name+'_shortcut')

    output = input
    if labels is not None:
        output = gan_cond_batch_norm(output, labels, n_labels=n_labels, name=name+'_bn1')

    output = tf.nn.relu(output)
    output = conv1(output, filter_size=filter_size, spectral_normed=spectral_normed,
                   update_collection=update_collection, name=name+'_conv1')

    if labels is not None:
        output = gan_cond_batch_norm(output, labels, n_labels=n_labels, name=name+'_bn2')

    output = tf.nn.relu(output)
    output = conv2(output, filter_size=filter_size, spectral_normed=spectral_normed,
                   update_collection=update_collection, name=name+'_conv2')

    return output + shortcut

def ResidualBlockDisc(input, dim_D, name, spectral_normed=False, update_collection=None):
    conv1 = partial(conv2d, output_dim=dim_D, spectral_normed=spectral_normed, update_collection=update_collection)
    conv2 = partial(ConvMeanPool, output_dim=dim_D, spectral_normed=spectral_normed, update_collection=update_collection)
    conv_shortcut = partial(MeanPoolConv, output_dim=dim_D, spectral_normed=spectral_normed, update_collection=update_collection)

    shortcut = conv_shortcut(input, filter_size=1, he_init=False, name=name+'_shortcut')
    output = input
    output = conv1(output, filter_size=3, name=name+'_conv1')
    output = tf.nn.relu(output)
    output = conv2(output, filter_size=3, name=name+'_conv2')
    return shortcut + output

