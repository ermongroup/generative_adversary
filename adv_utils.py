import tensorflow as tf
import os
import numpy as np


def label_smooth(y, weight=0.9):
    # requires y to be one_hot!
    return tf.clip_by_value(y, clip_value_min=(1.0 - weight) / (FLAGS.num_classes - 1.), clip_value_max=weight)


def random_flip_left_right(images):
    images_flipped = tf.reverse(images, axis=[2])
    flip = tf.cast(tf.contrib.distributions.Bernoulli(probs=tf.ones((tf.shape(images)[0],)) * 0.5).sample(), tf.bool)
    final_images = tf.where(flip, x=images, y=images_flipped)
    return final_images


def feature_squeeze(images, dataset='cifar'):
    # color depth reduction
    if dataset == 'cifar':
        npp = 2 ** 5
    elif dataset == 'mnist':
        npp = 2 ** 3

    npp_int = npp - 1
    images = images / 255.
    x_int = tf.rint(tf.multiply(images, npp_int))
    x_float = tf.div(x_int, npp_int)
    return median_filtering_2x2(x_float, dataset=dataset)


def median_filtering_2x2(images, dataset='cifar'):
    def median_filtering_layer_2x2(channel):
        top = tf.pad(channel, paddings=[[0, 0], [1, 0], [0, 0]], mode="REFLECT")[:, :-1, :]
        left = tf.pad(channel, paddings=[[0, 0], [0, 0], [1, 0]], mode="REFLECT")[:, :, :-1]
        top_left = tf.pad(channel, paddings=[[0, 0], [1, 0], [1, 0]], mode="REFLECT")[:, :-1, :-1]
        comb = tf.stack([channel, top, left, top_left], axis=3)
        return tf.nn.top_k(comb, 2).values[..., -1]

    if dataset == 'cifar':
        c0 = median_filtering_layer_2x2(images[..., 0])
        c1 = median_filtering_layer_2x2(images[..., 1])
        c2 = median_filtering_layer_2x2(images[..., 2])
        return tf.stack([c0, c1, c2], axis=3)
    elif dataset == 'mnist':
        return median_filtering_layer_2x2(images[..., 0])[..., None]


def normalize_image(images):
    return (images.astype(np.int32) - 127.5) / 127.5


def unnormalize_image(images):
    return images * 127.5 + 127.5

def get_weights_path(args):
    prefix = os.path.join('assets', 'pretrained')
    folder = args.dataset + '_' + args.classifier
    if args.trained:
        folder += '_trained'
    if args.adv:
        folder += '_adv'
    if args.adv_gen:
        folder += '_advgen'

    path = os.path.join(prefix, folder)
    if not os.path.exists(path):
        os.makedirs(path)

    ckpt_path = os.path.join(path, 'model.ckpt')
    return path, ckpt_path