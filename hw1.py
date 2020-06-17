import warnings
warnings.filterwarnings('ignore', category=FutureWarning) #消除Future Warning
import datetime
import io
import matplotlib.pyplot as plt

import os
import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds[:, -1, :, :], labels=labels[:, -1, :, :])
    loss = tf.reduce_mean(loss)
    #############################
    return loss


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        batch_size = input_images.shape[0]
        zero_labels = tf.zeros_like(input_labels[:, -1:, :, :])
        fixed_input_labels = tf.concat([input_labels[:, :-1, :, :], zero_labels], axis=1)
        input_dim = input_images.shape[-1]
        input = tf.concat((input_images, fixed_input_labels), -1)
        x = tf.reshape(input, [-1, self.samples_per_class*self.num_classes, input_dim + self.num_classes])
        x = self.layer1(x)
        x = self.layer2(x)
        out = tf.reshape(x, [-1 , self.samples_per_class, self.num_classes, self.num_classes])

        #############################
        return out

ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)

#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

with tf.Session() as sess:
    #writer = tf.summary.FileWriter('logs\\{}'.format(current_time), sess.graph)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    figure_train_loss, figure_test_loss , figure_acc , t= [], [], [], []

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            print("Test Accuracy", (1.0 * (pred == l)).mean())
            #writer.add_summary(merged)
            figure_train_loss.append(ls)
            figure_test_loss.append(tls)
            figure_acc.append(1.0 * (pred == l).mean())
            t.append(step)
    figure_train_loss_np = np.array(figure_train_loss)
    figure_test_loss_np = np.array(figure_test_loss)
    figure_acc_np = np.array(figure_acc)
    t_np = np.array(t)

    if not os.path.exists(".\\mylog"):
        os.mkdir(".\\mylog")
    np.save(".\\mylog\\{}-way-{}-shot_TrainLoss".format(FLAGS.num_classes, FLAGS.num_samples), figure_train_loss_np)
    np.save(".\\mylog\\{}-way-{}-shot_TestLoss".format(FLAGS.num_classes, FLAGS.num_samples), figure_test_loss_np)
    np.save(".\\mylog\\{}-way-{}-shot_acc".format(FLAGS.num_classes, FLAGS.num_samples), figure_acc_np)
    np.save(".\\mylog\\{}-way-{}-shot_step".format(FLAGS.num_classes, FLAGS.num_samples), t_np)


