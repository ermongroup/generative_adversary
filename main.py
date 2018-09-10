import time
import json
import models.resnet_model as resnet_model
from cleverhans.attacks_tf import fgm
from models.madry_mnist import MadryModel
from models.aditi_mnist import AditiMNIST
from models.zico_mnist import ZicoMNIST
from utils import *
from adv_utils import *
from models.vgg16 import vgg_16
from models.acwgan_gp import ACWGAN_GP
import argparse
from scipy.misc import imsave

parser = argparse.ArgumentParser("Generative Adversarial Examples")
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | celebA")
parser.add_argument('--adv', action='store_true', help="using adversarially trained network")
parser.add_argument('--classifier', type=str, default='resnet', help='resnet | vgg | madry | aditi | zico')
parser.add_argument('--datapath', type=str, default=None, help="input data path")
parser.add_argument('--seed', type=int, default=1234, help="random seed")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--mode', type=str, default='attack', help='eval | targeted_attack | untargeted_attack')
parser.add_argument('--top5', action='store_true', help="use top5 error")

parser.add_argument('--lr', type=float, default=1, help="learning rate for doing targeted/untargeted attack")
parser.add_argument('--n_adv_examples', type=int, default=1000000,
                    help="number of adversarial examples batches to search")
parser.add_argument('--n_iters', type=int, default=1000,
                    help="number of inner iterations for computing adversarial examples")
parser.add_argument('--z_dim', type=int, default=128, help="dimension of noise vector")
parser.add_argument('--checkpoint_dir', type=str, default='assets/checkpoint',
                    help='Directory name to save the checkpoints')
parser.add_argument('--result_dir', type=str, default='assets/results',
                    help='Directory name to save the generated images')
parser.add_argument('--log_dir', type=str, default='assets/logs',
                    help='Directory name to save training logs')
parser.add_argument('--source', type=int, default=0, help="ground truth class (source class)")
parser.add_argument('--target', type=int, default=1, help="target class")
parser.add_argument('--lambda1', type=float, default=100, help="coefficient for the closeness regularization term")
parser.add_argument('--lambda2', type=float, default=100, help="coefficient for the repulsive regularization term")
parser.add_argument('--n2collect', type=int, default=1024, help="number of adversarial examples to collect")
parser.add_argument('--eps', type=float, default=0.1, help="eps for attack augmented with noise")
parser.add_argument('--noise', action="store_true", help="add noise augmentation to attacks")
parser.add_argument('--z_eps', type=float, default=0.1, help="soft constraint for the search region of latent space")
parser.add_argument('--adv_gen', action="store_true", help="adversarial training using generative adversarial examples")
parser.add_argument('--trained', action="store_true", help="trained models")

args = parser.parse_args()


def resnet_template(images, training, hps):
    # Do per image standardization
    images_standardized = per_image_standardization(images)
    model = resnet_model.ResNet(hps, images_standardized, training)
    model.build_graph()
    return model.logits


def vgg_template(images, training, hps):
    images_standardized = per_image_standardization(images)
    logits, _ = vgg_16(images_standardized, num_classes=hps.num_classes, is_training=training, dataset=hps.dataset)
    return logits


def madry_template(images, training):
    model = MadryModel(images)
    return model.pre_softmax


def aditi_template(images, training):
    model = AditiMNIST(images)
    return model.logits


def zico_template(images, training):
    model = ZicoMNIST(images)
    return model.logits


def evaluate(hps, data_X, data_y, eval_once=True):
    """Eval loop."""
    images = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, args.channels))

    labels_onehot = tf.placeholder(tf.int32, shape=(None, args.num_classes))
    labels = tf.argmax(labels_onehot, axis=1)

    if args.classifier == "madry":
        net = tf.make_template('net', madry_template)
        logits = net(images, training=False)
    elif args.classifier == 'aditi':
        net = tf.make_template('net', aditi_template)
        logits = net(images, training=False)
    elif args.classifier == 'zico':
        net = tf.make_template('net', zico_template)
        logits = net(images, training=False)
    else:
        net = tf.make_template('net', resnet_template, hps=hps) if args.classifier == 'resnet' else \
            tf.make_template('net', vgg_template, hps=hps)
        logits = net(images, training=False)

    pred = tf.argmax(logits, axis=1)
    probs = tf.nn.softmax(logits)

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_onehot)
    adv_image = fgm(images, tf.nn.softmax(logits), y=labels_onehot, eps=args.eps / 10, clip_min=0.0, clip_max=1.0)
    top_5 = tf.nn.in_top_k(predictions=logits, targets=labels, k=5)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net'))
    if args.classifier == 'madry' and not args.trained:
        saver = tf.train.Saver(
            {x.name[4:-2]: x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net")})

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    best_precision = 0.0
    save_path, save_path_ckpt = get_weights_path(args)
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(save_path)
        except tf.errors.OutOfRangeError as e:
            print('[!] Cannot restore checkpoint: %s', e)
            break
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('[!] No model to eval yet at %s', save_path)
            break
        print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        adv_prediction = 0
        total_loss = 0
        all_preds = []
        batch_size = args.batch_size
        num_batch = len(data_X) // batch_size
        bad_images = []
        bad_labels = []
        confidences = []
        adv_images = []
        cls_preds = []
        true_labels = []
        for batch in range(num_batch):
            x = data_X[batch * batch_size: (batch + 1) * batch_size]
            x = x.astype(np.float32)
            y = data_y[batch * batch_size: (batch + 1) * batch_size]
            y = y.astype(np.int32)
            if not args.top5:
                (loss, predictions, conf) = sess.run(
                    [cost, pred, probs], feed_dict={
                        images: x,
                        labels_onehot: y
                    })
                all_preds.extend(predictions)
                confidences.extend(conf[np.arange(conf.shape[0]), predictions])
                img_np = np.copy(x)
                for i in range(100):
                    img_np = sess.run(adv_image, feed_dict={
                        images: img_np,
                        labels_onehot: y
                    })
                    img_np = np.clip(img_np, x - args.eps, x + args.eps)
                    img_np = np.clip(img_np, 0.0, 1.0)
                adv_images.extend(img_np)

                adv_pred_np = pred.eval(session=sess, feed_dict={
                    images: img_np,
                    labels_onehot: y
                })

                cls_preds.extend(adv_pred_np)
                true_labels.extend(np.argmax(y, axis=1))
            else:
                (loss, in_top5) = sess.run(
                    [cost, top_5], feed_dict={
                        images: x,
                        labels_onehot: y
                    }
                )
            total_loss += np.sum(loss)
            y = np.argmax(y, axis=1)
            correct_prediction += np.sum(y == predictions) if not args.top5 else np.sum(in_top5)
            bad_images.extend(x[y != predictions])
            bad_labels.extend(predictions[y != predictions])
            adv_prediction += np.sum(y == adv_pred_np)
            total_prediction += loss.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        loss = 1.0 * total_loss / total_prediction
        best_precision = max(precision, best_precision)
        average_conf = np.mean(np.asarray(confidences))
        adv_images = np.asarray(adv_images)
        cls_preds = np.asarray(cls_preds)
        true_labels = np.asarray(true_labels)

        if not args.top5:
            print('[*] loss: %.6f, precision: %.6f, PGD precision: %.6f, Confidence: %.6f' %
                  (loss, precision, adv_prediction / total_prediction, average_conf))
            folder_format = '/atlas/u/yangsong/generative_adversary/{}_{}_pgd/'
            np.savez(os.path.join(check_folder(folder_format.format(args.dataset, args.classifier)),
                                  'eps_{:.3f}.npz'.format(args.eps)),
                     adv_images=adv_images, cls_preds=cls_preds, true_labels=true_labels)
        else:
            print('[*] loss: %.6f, top 5 accuracy: %.6f, best top 5 accuracy: %.6f' %
                  (loss, precision, best_precision))

        bad_images = np.asarray(bad_images)
        bad_images = (255. * bad_images).astype(np.uint8)
        bad_labels = np.asarray(bad_labels).astype(np.uint8)

        if len(bad_images) > 10:
            bad_images = bad_images[:10]
            bad_labels = bad_labels[:10]

        bad_images = np.reshape(bad_images, (len(bad_images) * args.image_size, args.image_size, args.channels))
        bad_images = np.squeeze(bad_images)

        imsave(os.path.join(check_folder('tmp'), 'bad_images.png'), bad_images)
        print("bad_labels:\n{}".format(bad_labels))

        if eval_once:
            break

        time.sleep(60)


def untargeted_attack(hps, lambda1, lambda2, source, noise=False):
    """generative adversarial attack"""

    source_np = np.asarray([source] * args.batch_size).astype(np.int32)
    if args.classifier == "madry":
        net = tf.make_template('net', madry_template)
    elif args.classifier == 'aditi':
        net = tf.make_template('net', aditi_template)
    elif args.classifier == 'zico':
        net = tf.make_template('net', zico_template)
    else:
        net = tf.make_template('net', resnet_template, hps=hps) if args.classifier == 'resnet' else \
            tf.make_template('net', vgg_template, hps=hps)

    adv_noise = tf.get_variable('adv_noise', shape=(args.batch_size, args.image_size, args.image_size, args.channels),
                                dtype=tf.float32, initializer=tf.zeros_initializer)
    adv_z = tf.get_variable('adv_z',
                            shape=(args.batch_size, args.z_dim),
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer)

    ref_z = tf.get_variable('ref_z',
                            shape=(args.batch_size, args.z_dim),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if args.dataset == 'mnist':
        dim_D = 32
        dim_G = 32
    elif args.dataset == 'svhn':
        dim_D = 128
        dim_G = 128
    elif args.dataset == 'celebA':
        dim_D = 64
        dim_G = 64

    acgan = ACWGAN_GP(
        sess,
        epoch=10,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        result_dir=args.result_dir,
        log_dir=args.log_dir,
        dim_D=dim_D,
        dim_G=dim_G
    )

    acgan.build_model()

    adv_images = acgan.generator(adv_z, source_np, reuse=True)

    _, acgan_logits = acgan.discriminator(adv_images, update_collection=None, reuse=True)
    acgan_pred = tf.argmax(acgan_logits, axis=1)

    if noise:
        adv_images += args.eps * tf.nn.tanh(adv_noise)
        if args.dataset in ('svhn', 'celebA'):
            adv_images = tf.clip_by_value(adv_images, clip_value_min=-1., clip_value_max=1.0)
        else:
            adv_images = tf.clip_by_value(adv_images, clip_value_min=0., clip_value_max=1.)

    net_logits = net(adv_images, training=False)
    net_softmax = tf.nn.softmax(net_logits)
    net_pred = tf.argmax(net_logits, axis=1)

    # loop over all classes
    obj_classes = []
    for i in range(args.num_classes):
        if i == source:
            continue
        onehot = np.zeros((args.batch_size, args.num_classes), dtype=np.float32)
        onehot[:, i] = 1.0
        obj_classes.append(tf.nn.softmax_cross_entropy_with_logits(logits=net_logits, labels=onehot))

    all_cross_entropy = tf.stack(obj_classes, axis=1)
    min_cross_entropy = tf.reduce_mean(tf.reduce_min(all_cross_entropy, axis=1))

    obj = min_cross_entropy + \
          lambda1 * tf.reduce_mean(tf.maximum(tf.square(ref_z - adv_z) - args.z_eps ** 2, 0.0)) + \
          lambda2 * tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acgan_logits, labels=source_np))

    _iter = tf.placeholder(tf.float32, shape=(), name="iter")
    with tf.variable_scope("train_ops"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
        var = 0.01 / (1. + _iter) ** 0.55
        if noise:
            grads = optimizer.compute_gradients(obj, var_list=[adv_z, adv_noise])
        else:
            grads = optimizer.compute_gradients(obj, var_list=[adv_z])

        new_grads = []
        for grad, v in grads:
            if v is not adv_noise:
                new_grads.append((grad + tf.random_normal(shape=grad.get_shape().as_list(), stddev=tf.sqrt(var)), v))
            else:
                new_grads.append((grad / tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3], keep_dims=True)), v))

        adv_op = optimizer.apply_gradients(new_grads)

    momentum_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_ops'))
    init_op = tf.group(momentum_init, tf.variables_initializer([adv_z, adv_noise]))
    with tf.control_dependencies([init_op]):
        init_op = tf.group(init_op, tf.assign(ref_z, adv_z))

    sess.run(tf.global_variables_initializer())
    # load classifier
    save_path, save_path_ckpt = get_weights_path(args)
    if args.classifier == 'madry':
        if args.trained:
            saver4classifier = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net"))
        else:
            saver4classifier = tf.train.Saver(
                {x.name[4:-2]: x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net")})
    else:
        saver4classifier = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net'))

    checkpoint_dir = os.path.join(args.checkpoint_dir, acgan.model_dir, acgan.model_name)
    try:
        ckpt_state = tf.train.get_checkpoint_state(save_path)
    except tf.errors.OutOfRangeError as e:
        print('[!] Cannot restore checkpoint: %s' % e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        print('[!] No model to eval yet at %s' % save_path)
    print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    saver4classifier.restore(sess, ckpt_state.model_checkpoint_path)
    # load ACGAN
    saver4gen = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='generator|discriminator|classifier'))
    try:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
    except tf.errors.OutOfRangeError as e:
        print('[!] Cannot restore checkpoint: %s' % e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        print('[!] No model to eval yet at %s' % checkpoint_dir)
    print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    saver4gen.restore(sess, ckpt_state.model_checkpoint_path)

    acc = 0.
    adv_acc = 0.
    adv_im_np = []
    adv_labels_np = []
    latent_z = []
    for batch in range(args.n_adv_examples):
        # random ground truth classes
        sess.run(init_op)
        preds_np, probs_np, im_np, cost_before = sess.run([net_pred, net_softmax, adv_images, obj])

        ###### Using GD for attacking
        # initialize optimizers
        for i in range(args.n_iters):
            _, now_cost, pred_np, acgan_pred_np = sess.run([adv_op, obj, net_pred, acgan_pred], feed_dict={_iter: i})
            ok = np.logical_and(pred_np != source, acgan_pred_np == source)
            print("   [*] {}th iter, cost: {}, success: {}/{}".format(i + 1, now_cost, np.sum(ok), args.batch_size))

        adv_preds_np, adv_probs_np, im_np, cost_after, hidden_z, acgan_pred_np = sess.run(
            [net_pred, net_softmax, adv_images, obj, adv_z, acgan_pred])
        acc += np.sum(preds_np == source)
        adv_acc += np.sum(adv_preds_np == source)
        idx = np.logical_and(adv_preds_np != source, acgan_pred_np == source)
        adv_im_np.extend(im_np[idx])
        adv_labels_np.extend(adv_preds_np[idx])
        latent_z.extend(hidden_z[idx])
        print("batch: {}, acc: {}, adv_acc: {}, num collected: {}, cost before: {}, cost after: {}".
              format(batch + 1,
                     acc / ((batch + 1) * args.batch_size),
                     adv_acc / ((batch + 1) * args.batch_size),
                     len(adv_im_np), cost_before, cost_after))

        if len(adv_im_np) >= args.n2collect:
            adv_im_np = np.asarray(adv_im_np)
            adv_labels_np = np.asarray(adv_labels_np)
            latent_z = np.asarray(latent_z)
            classifier = args.classifier
            if args.adv:
                classifier += '_adv'

            folder_format = '{}_{}_untargeted_attack'
            if noise: folder_format += '_noise'
            np.savez(os.path.join(check_folder(folder_format.format(args.dataset, classifier)),
                                  'source_{}'.format(source)), adv_labels=adv_labels_np, adv_imgs=adv_im_np,
                     latent_z=latent_z)
            size = int(np.sqrt(args.n2collect))
            write_labels(adv_labels_np, args.dataset, size)
            img = label_images(adv_im_np[:args.n2collect, ...], adv_labels_np[:args.n2collect])
            save_images(img, [size, size], os.path.join(
                check_folder(folder_format.format(args.dataset, classifier)), 's_{}_ims.png').format(args.source))
            break


def targeted_attack(hps, source, target, lambda1, lambda2, noise=False):
    """targeted generative adversarial attack"""

    source_np = np.asarray([source] * args.batch_size).astype(np.int32)
    target_np = np.asarray([target] * args.batch_size).astype(np.int32)

    if args.classifier == "madry":
        net = tf.make_template('net', madry_template)
    elif args.classifier == 'aditi':
        net = tf.make_template('net', aditi_template)
    elif args.classifier == 'zico':
        net = tf.make_template('net', zico_template)
    else:
        net = tf.make_template('net', resnet_template, hps=hps) if args.classifier == 'resnet' else \
            tf.make_template('net', vgg_template, hps=hps)

    adv_noise = tf.get_variable('adv_noise', shape=(args.batch_size, args.image_size, args.image_size, args.channels),
                                dtype=tf.float32, initializer=tf.zeros_initializer)
    adv_z = tf.get_variable('adv_z',
                            shape=(args.batch_size, args.z_dim),
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer)

    ref_z = tf.get_variable('ref_z',
                            shape=(args.batch_size, args.z_dim),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if args.dataset == 'mnist':
        dim_D = 32
        dim_G = 32
    elif args.dataset == 'svhn':
        dim_D = 128
        dim_G = 128
    elif args.dataset == 'celebA':
        dim_D = 64
        dim_G = 64

    acgan = ACWGAN_GP(
        sess,
        epoch=10,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        result_dir=args.result_dir,
        log_dir=args.log_dir,
        dim_D=dim_D,
        dim_G=dim_G
    )

    acgan.build_model()

    adv_images = acgan.generator(adv_z, source_np, reuse=True)
    _, acgan_logits = acgan.discriminator(adv_images, update_collection=None, reuse=True)
    acgan_pred = tf.argmax(acgan_logits, axis=1)
    acgan_softmax = tf.nn.softmax(acgan_logits)

    if noise:
        adv_images += args.eps * tf.tanh(adv_noise)
        if args.dataset in ('svhn', 'celebA'):
            adv_images = tf.clip_by_value(adv_images, clip_value_min=-1., clip_value_max=1.)
        else:
            adv_images = tf.clip_by_value(adv_images, clip_value_min=0., clip_value_max=1.0)

    net_logits = net(adv_images, training=False)
    net_softmax = tf.nn.softmax(net_logits)
    net_pred = tf.argmax(net_logits, axis=1)

    obj = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net_logits, labels=target_np)) + \
          lambda1 * tf.reduce_mean(tf.maximum(tf.square(ref_z - adv_z) - args.z_eps ** 2, 0.0)) + \
          lambda2 * tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acgan_logits, labels=source_np))

    _iter = tf.placeholder(tf.float32, shape=(), name="iter")
    with tf.variable_scope("train_ops"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
        var = 0.01 / (1. + _iter) ** 0.55
        if noise:
            grads = optimizer.compute_gradients(obj, var_list=[adv_z, adv_noise])
        else:
            grads = optimizer.compute_gradients(obj, var_list=[adv_z])

        new_grads = []
        for grad, v in grads:
            if v is not adv_noise:
                new_grads.append((grad + tf.random_normal(shape=grad.get_shape().as_list(), stddev=tf.sqrt(var)), v))
            else:
                new_grads.append((grad / tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3], keep_dims=True)), v))

        adv_op = optimizer.apply_gradients(new_grads)

    momentum_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_ops'))
    init_op = tf.group(momentum_init, tf.variables_initializer([adv_z, adv_noise]))
    with tf.control_dependencies([init_op]):
        init_op = tf.group(init_op, tf.assign(ref_z, adv_z))

    sess.run(tf.global_variables_initializer())
    # load classifier
    save_path, save_path_ckpt = get_weights_path(args)
    if args.classifier == 'madry':
        if args.trained:
            saver4classifier = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net"))
        else:
            saver4classifier = tf.train.Saver(
                {x.name[4:-2]: x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net")})
    else:
        saver4classifier = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net'))
    checkpoint_dir = os.path.join(args.checkpoint_dir, acgan.model_dir, acgan.model_name)
    try:
        ckpt_state = tf.train.get_checkpoint_state(save_path)
    except tf.errors.OutOfRangeError as e:
        print('[!] Cannot restore checkpoint: %s' % e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        print('[!] No model to eval yet at %s' % save_path)
    print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    saver4classifier.restore(sess, ckpt_state.model_checkpoint_path)
    # load ACGAN

    saver4gen = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='generator|discriminator|classifier'))
    try:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
    except tf.errors.OutOfRangeError as e:
        print('[!] Cannot restore checkpoint: %s' % e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        print('[!] No model to eval yet at %s' % checkpoint_dir)
    print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    saver4gen.restore(sess, ckpt_state.model_checkpoint_path)

    acc = 0.
    adv_acc = 0.
    adv_im_np = []
    latent_z = []
    init_latent_z = []
    for batch in range(args.n_adv_examples):
        ###### random ground truth classes
        sess.run(init_op)
        preds_np, probs_np, im_np, cost_before = sess.run([net_pred, net_softmax, adv_images, obj])

        ###### Using GD for attacking
        # initialize optimizers
        for i in range(args.n_iters):
            _, now_cost, pred_np, acgan_pred_np, acgan_probs = sess.run(
                [adv_op, obj, net_pred, acgan_pred, acgan_softmax],
                feed_dict={_iter: i})
            ok = np.logical_and(pred_np == target, acgan_pred_np == source)
            print("   [*] {}th iter, cost: {}, success: {}/{}".format(i + 1, now_cost, np.sum(ok), args.batch_size))

        adv_preds_np, acgan_preds_np, adv_probs_np, acgan_probs_np, im_np, hidden_z, init_z, cost_after = sess.run(
            [net_pred, acgan_pred,
             net_softmax, acgan_softmax, adv_images, adv_z, ref_z, obj])
        acc += np.sum(preds_np == source)
        idx = np.logical_and(adv_preds_np == target, acgan_preds_np == source)
        adv_acc += np.sum(idx)
        adv_im_np.extend(im_np[idx])
        latent_z.extend(hidden_z[idx])
        init_latent_z.extend(init_z[idx])
        print("batch: {}, acc: {}, adv_acc: {}, num collected: {}, cost before: {}, cost after: {}".
              format(batch + 1, acc / ((batch + 1) * args.batch_size), adv_acc / ((batch + 1) * args.batch_size),
                     len(adv_im_np), cost_before, cost_after))

        if len(adv_im_np) >= args.n2collect:
            adv_im_np = np.asarray(adv_im_np)
            latent_z = np.asarray(latent_z)
            size = int(np.sqrt(args.n2collect))
            classifier = args.classifier
            if args.adv:
                classifier += '_adv'

            folder_format = '{}_{}_targeted_attack_with_z0'
            if noise: folder_format += '_noise'
            np.savez(os.path.join(check_folder(folder_format.format(args.dataset, classifier)),
                                  'from{}to{}'.format(source, target)), adv_imgs=adv_im_np, latent_z=latent_z,
                     init_latent_z=init_latent_z)
            save_images(adv_im_np[:args.n2collect, :, :, :], [size, size],
                        os.path.join(check_folder(folder_format.format(args.dataset, classifier)),
                                     '{}_ims_from{}_to{}.png').format(args.dataset, source, target))
            break


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.dataset == 'mnist':
        num_classes = 10
        args.num_classes = 10
        args.image_size = 28
        args.channels = 1

        if args.mode == 'eval':
            data_X, data_y, test_X, test_y = load_mnist4classifier(args.dataset)

    elif args.dataset == 'svhn':
        num_classes = 10
        args.num_classes = 10
        args.image_size = 32
        args.channels = 3

        if args.mode == 'eval':
            data_X, data_y, test_X, test_y = load_svhn4classifier()

    elif args.dataset == 'celebA':
        num_classes = 2
        args.num_classes = 2
        args.image_size = 64
        args.channels = 3

        if args.mode == 'eval':
            data_X, data_y, test_X, test_y = load_celebA4classifier()

    else:
        raise NotImplementedError("Dataset {} not supported!".format(args.dataset))

    print("[*] input args:\n", json.dumps(vars(args), indent=4, separators=(',', ':')))

    num_residual_units = 5
    hps = resnet_model.HParams(batch_size=args.batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=num_residual_units,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom',
                               dataset=args.dataset)

    if args.mode == 'eval':
        evaluate(hps, test_X, test_y)
    elif args.mode == 'targeted_attack':
        targeted_attack(hps, args.source, args.target, args.lambda1, args.lambda2, noise=args.noise)
    elif args.mode == 'untargeted_attack':
        untargeted_attack(hps, args.lambda1, args.lambda2, source=args.source, noise=args.noise)
    else:
        raise NotImplementedError("No modes other than eval and attack!")


if __name__ == '__main__':
    main()
