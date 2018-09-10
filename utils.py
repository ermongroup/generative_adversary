"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import scipy.misc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os, gzip
import cv2 as cv

import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_mnist(dataset_name, trainonly=False):
    data_dir = os.path.join("assets/data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    if trainonly:
        X = trX
        y = trY.astype(np.int)
    else:
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def load_svhn(source_class=None, trainonly=False):
    print("[*] Loading SVHN")
    data_dir = os.path.join("assets", "data", "svhn")

    def extract_data(filename):
        data = sio.loadmat(os.path.join(data_dir, filename))
        X = data['X'].transpose(3, 0, 1, 2)
        y = data['y'].reshape((-1))
        y[y == 10] = 0
        return X, y.astype(np.int)

    trX, trY = extract_data('train_32x32.mat')
    teX, teY = extract_data('test_32x32.mat')
    exX, exY = extract_data('extra_32x32.mat')

    print("[*] SVHN loaded")

    if trainonly:
        X = trX
        y = trY
    else:
        X = np.concatenate([trX, teX, exX], axis=0)
        y = np.concatenate([trY, teY, exY], axis=0)

    if source_class is not None:
        idx = (y == source_class)
        X = X[idx]
        y = y[idx]

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    y_vec[np.arange(0, len(y)), y] = 1.0
    return X / 255., y_vec


def load_celebA():
    print("[*] Loading CelebA")
    X = sio.loadmat('/atlas/u/ruishu/data/celeba64_zoom.mat')['images']
    y = sio.loadmat('/atlas/u/ruishu/data/celeba_gender.mat')['y']
    y = np.eye(2)[y.reshape(-1)]

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    return X / 255., y


def load_celebA4classifier():
    print("[*] Loading CelebA")
    X = sio.loadmat('/atlas/u/ruishu/data/celeba64_zoom.mat')['images']
    y = sio.loadmat('/atlas/u/ruishu/data/celeba_gender.mat')['y']
    y = np.eye(2)[y.reshape(-1)]

    trX = X[:150000]
    trY = y[:150000]
    teX = X[150000:]
    teY = y[150000:]
    return trX / 255., trY, teX / 255., teY


def load_svhn4classifier():
    print("[*] Loading SVHN")
    data_dir = os.path.join("assets", "data", "svhn")

    def extract_data(filename):
        data = sio.loadmat(os.path.join(data_dir, filename))
        X = data['X'].transpose(3, 0, 1, 2)
        y = data['y'].reshape((-1))
        y[y == 10] = 0
        return X, y.astype(np.int)

    trX, trY = extract_data('train_32x32.mat')
    teX, teY = extract_data('test_32x32.mat')
    print("[*] SVHN loaded")
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)

    tr_y_vec = np.zeros((len(trY), 10), dtype=np.float)
    tr_y_vec[np.arange(0, len(trY)), trY] = 1.0

    te_y_vec = np.zeros((len(teY), 10), dtype=np.float)
    te_y_vec[np.arange(0, len(teY)), teY] = 1.0
    return trX / 255., tr_y_vec, teX / 255., te_y_vec


def load_mnist4classifier(dataset_name):
    data_dir = os.path.join("assets/data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)

    tr_y_vec = np.zeros((len(trY), 10), dtype=np.float)
    tr_y_vec[np.arange(0, len(trY)), trY] = 1.0
    te_y_vec = np.zeros((len(teY), 10), dtype=np.float)
    te_y_vec[np.arange(0, len(teY)), teY] = 1.0

    return trX / 255., tr_y_vec, teX / 255., te_y_vec


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def write_labels(labels, dataset, size):
    if dataset in ('mnist', 'svhn'):
        dic = {x: str(x) for x in range(10)}
    else:
        raise NotImplementedError("Dataset {} not supported".format(dataset))
    print("adversarial labels:")
    for i in range(size):
        for j in range(size):
            print("{}".format(dic[labels[i * size + j]]), end='\t')
        print("")


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def label_images(images, labels):
    font = cv.FONT_HERSHEY_SIMPLEX
    new_imgs = []
    for i, img in enumerate(images):
        new_img = ((img.copy() + 1.) * 127.5).astype(np.uint8)
        if new_img.shape[-1] == 3:
            new_img = new_img[..., ::-1]
            new_img = cv.resize(new_img, (100, 100), interpolation=cv.INTER_LINEAR)
            new_img = cv.putText(new_img, str(labels[i]), (10, 30), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            new_img = cv.copyMakeBorder(new_img, top=2, bottom=2, left=2, right=2, borderType=cv.BORDER_CONSTANT,
                                        value=(255, 255, 255))
        else:
            new_img = np.squeeze(new_img)
            new_img = cv.resize(new_img, (100, 100), interpolation=cv.INTER_LINEAR)
            new_img = cv.putText(new_img, str(labels[i]), (10, 30), font, 1, (255), 2, cv.LINE_AA)
            new_img = new_img[..., None]

        new_img = (new_img / 127.5 - 1.0).astype(np.float32)
        new_imgs.append(new_img[..., ::-1])
    return np.stack(new_imgs, axis=0)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


""" Drawing Tools """


# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def per_image_standardization(images, image_size=28):
    image_mean, image_std = tf.nn.moments(images, axes=[1, 2, 3])
    image_std = tf.sqrt(image_std)[:, None, None, None]
    images_standardized = (images - image_mean[:, None, None, None]) / tf.maximum(image_std, 1.0 / np.sqrt(
        image_size ** 2 * 3))
    return images_standardized


def gradients(f, x, grad_ys=None):
    '''
    An easier way of computing gradients in tensorflow. The difference from tf.gradients is
        * If f is not connected with x in the graph, it will output 0s instead of Nones. This will be more meaningful
            for computing higher-order gradients.

        * The output will have the same shape and type as x. If x is a list, it will be a list. If x is a Tensor, it
            will be a tensor as well.

    :param f: A `Tensor` or a list of tensors to be differentiated
    :param x: A `Tensor` or a list of tensors to be used for differentiation
    :param grad_ys: Optional. It is a `Tensor` or a list of tensors having exactly the same shape and type as `f` and
                    holds gradients computed for each of `f`.
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`
    '''

    if isinstance(x, list):
        grad = tf.gradients(f, x, grad_ys=grad_ys)
        for i in range(len(x)):
            if grad[i] is None:
                grad[i] = tf.zeros_like(x[i])
        return grad
    else:
        grad = tf.gradients(f, x, grad_ys=grad_ys)[0]
        if grad is None:
            return tf.zeros_like(x)
        else:
            return grad


def Lop(f, x, v):
    '''
    Compute Jacobian-vector product. The result is v^T @ J_x

    :param f: A `Tensor` or a list of tensors for computing the Jacobian J_x
    :param x: A `Tensor` or a list of tensors with respect to which the Jacobian is computed.
    :param v: A `Tensor` or a list of tensors having the same shape and type as `f`
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`
    '''
    assert not isinstance(f, list) or isinstance(v, list), "f and v should be of the same type"
    return gradients(f, x, grad_ys=v)


def Rop(f, x, v):
    '''
    Compute Jacobian-vector product. The result is J_x @ v.
    The method is inspired by [deep yearning's blog](https://j-towns.github.io/2017/06/12/A-new-trick.html)
    :param f: A `Tensor` or a list of tensors for computing the Jacobian J_x
    :param x: A `Tensor` or a list of tensors with respect to which the Jacobian is computed
    :param v: A `Tensor` or a list of tensors having the same shape and type as `v`
    :return: A `Tensor` or a list of tensors having the same shape and type as `f`
    '''
    assert not isinstance(x, list) or isinstance(v, list), "x and v should be of the same type"
    if isinstance(f, list):
        w = [tf.ones_like(_) for _ in f]
    else:
        w = tf.ones_like(f)
    return gradients(Lop(f, x, w), w, grad_ys=v)
